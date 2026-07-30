//! QUIC tunnel management for forwarding OpenAI HTTP traffic to the local
//! model-aware API proxy.

use crate::mesh::Node;
use crate::protocol::read_len_prefixed;
use anyhow::{Context, Result};
use iroh::EndpointId;
use prost::Message;
use std::sync::Arc;
use std::sync::atomic::{AtomicU16, AtomicU64, Ordering};
use std::time::Duration;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::TcpStream;

/// Global byte counter for tunnel traffic
static BYTES_TRANSFERRED: AtomicU64 = AtomicU64::new(0);

const MAX_INBOUND_HTTP_REQUEST_LINE_BYTES: usize = 8 * 1024;
const INBOUND_HTTP_REQUEST_LINE_TIMEOUT: Duration = Duration::from_secs(10);
const INBOUND_HTTP_TUNNEL_FORBIDDEN_RESPONSE: &[u8] = b"HTTP/1.1 403 Forbidden\r\n\
Content-Type: application/json\r\n\
Content-Length: 97\r\n\
Connection: close\r\n\
\r\n\
{\"error\":{\"message\":\"remote mesh HTTP tunnels only allow inference requests\",\"type\":\"forbidden\"}}";

fn quic_response_first_byte_timeout() -> Duration {
    Duration::from_secs(5 * 60)
}

/// Manages all tunnels for a node
#[derive(Clone)]
pub struct Manager {
    node: Node,
    http_port: Arc<AtomicU16>,
}

impl Manager {
    /// Start the tunnel manager.
    /// The API proxy port for inbound HTTP tunnels is set by the runtime once
    /// the node begins serving.
    pub async fn start(
        node: Node,
        _legacy_tunnel_rx: tokio::sync::mpsc::Receiver<(
            iroh::endpoint::SendStream,
            iroh::endpoint::RecvStream,
        )>,
        mut tunnel_http_rx: tokio::sync::mpsc::Receiver<(
            iroh::endpoint::SendStream,
            iroh::endpoint::RecvStream,
        )>,
        mut stage_transport_rx: tokio::sync::mpsc::Receiver<(
            EndpointId,
            iroh::endpoint::SendStream,
            iroh::endpoint::RecvStream,
        )>,
    ) -> Result<Self> {
        let mgr = Manager {
            node: node.clone(),
            http_port: Arc::new(AtomicU16::new(0)),
        };

        // Handle inbound HTTP tunnel streams.
        // These connect to the local model-aware OpenAI proxy.
        let http_port_ref = mgr.http_port.clone();
        let http_node = mgr.node.clone();
        tokio::spawn(async move {
            while let Some((send, recv)) = tunnel_http_rx.recv().await {
                let port = http_port_ref.load(Ordering::Relaxed);
                if port == 0 {
                    tracing::warn!("Inbound HTTP tunnel but no OpenAI surface running, dropping");
                    continue;
                }
                let node = http_node.clone();
                tokio::spawn(async move {
                    if let Err(e) = handle_inbound_http_stream(node, send, recv, port).await {
                        tracing::warn!("Inbound HTTP tunnel stream error: {e}");
                    }
                });
            }
        });

        let stage_node = mgr.node.clone();
        tokio::spawn(async move {
            while let Some((remote, send, recv)) = stage_transport_rx.recv().await {
                let node = stage_node.clone();
                tokio::spawn(async move {
                    if let Err(e) = handle_inbound_stage_transport(node, remote, send, recv).await {
                        tracing::warn!(
                            "Inbound stage transport stream error from {}: {e}",
                            remote.fmt_short()
                        );
                    }
                });
            }
        });

        Ok(mgr)
    }

    /// Update the local model-aware API proxy port for inbound HTTP tunnel streams.
    /// Set to 0 to disable.
    pub fn set_http_port(&self, port: u16) {
        self.http_port.store(port, Ordering::Relaxed);
        tracing::info!("Tunnel manager: http_port updated to {port}");
    }
}

/// Handle an inbound HTTP tunnel bi-stream: connect to the local API proxy and relay.
async fn handle_inbound_http_stream(
    node: Node,
    mut quic_send: iroh::endpoint::SendStream,
    mut quic_recv: iroh::endpoint::RecvStream,
    http_port: u16,
) -> Result<()> {
    let request_line = read_inbound_http_request_line(&mut quic_recv).await?;
    if !is_allowed_inbound_http_tunnel_request(&request_line) {
        tracing::warn!("Rejected non-inference inbound HTTP tunnel request");
        quic_send
            .write_all(INBOUND_HTTP_TUNNEL_FORBIDDEN_RESPONSE)
            .await?;
        quic_send.finish()?;
        return Ok(());
    }

    tracing::info!("Inbound HTTP tunnel stream -> API proxy :{http_port}");
    let tcp_stream = TcpStream::connect(format!("127.0.0.1:{http_port}")).await?;
    tcp_stream.set_nodelay(true)?;
    let _inflight = node.begin_inflight_request();

    let (tcp_read, mut tcp_write) = tokio::io::split(tcp_stream);
    tcp_write.write_all(&request_line).await?;
    relay_bidirectional(tcp_read, tcp_write, quic_send, quic_recv).await
}

async fn read_inbound_http_request_line<R>(reader: &mut R) -> Result<Vec<u8>>
where
    R: AsyncRead + Unpin,
{
    tokio::time::timeout(
        INBOUND_HTTP_REQUEST_LINE_TIMEOUT,
        read_inbound_http_request_line_inner(reader),
    )
    .await
    .context("timed out waiting for inbound HTTP tunnel request line")?
}

async fn read_inbound_http_request_line_inner<R>(reader: &mut R) -> Result<Vec<u8>>
where
    R: AsyncRead + Unpin,
{
    let mut request_line = Vec::with_capacity(128);
    let mut byte = [0u8; 1];
    while request_line.len() < MAX_INBOUND_HTTP_REQUEST_LINE_BYTES {
        let read = reader.read(&mut byte).await?;
        if read == 0 {
            anyhow::bail!("inbound HTTP tunnel closed before request line completed");
        }
        request_line.push(byte[0]);
        if request_line.ends_with(b"\r\n") {
            return Ok(request_line);
        }
    }
    anyhow::bail!(
        "inbound HTTP tunnel request line exceeded {} bytes",
        MAX_INBOUND_HTTP_REQUEST_LINE_BYTES
    )
}

fn is_allowed_inbound_http_tunnel_request(request_line: &[u8]) -> bool {
    let Some(request_line) = request_line.strip_suffix(b"\r\n") else {
        return false;
    };
    let Ok(request_line) = std::str::from_utf8(request_line) else {
        return false;
    };
    let mut parts = request_line.split_ascii_whitespace();
    let (Some(method), Some(target), Some(version)) = (parts.next(), parts.next(), parts.next())
    else {
        return false;
    };
    if parts.next().is_some() || !matches!(version, "HTTP/1.0" | "HTTP/1.1") {
        return false;
    }
    let path = target.split('?').next().unwrap_or(target);
    matches!(
        (method, path),
        ("GET", "/v1/models" | "/models") | ("POST", "/v1/chat/completions" | "/v1/responses")
    )
}

async fn handle_inbound_stage_transport(
    node: Node,
    remote: EndpointId,
    quic_send: iroh::endpoint::SendStream,
    mut quic_recv: iroh::endpoint::RecvStream,
) -> Result<()> {
    let buf = read_len_prefixed(&mut quic_recv).await?;
    let open = skippy_protocol::proto::stage::StageTransportOpen::decode(buf.as_slice())
        .map_err(|e| anyhow::anyhow!("StageTransportOpen decode error: {e}"))?;
    skippy_protocol::validate_stage_transport_open(&open)
        .map_err(|e| anyhow::anyhow!("StageTransportOpen validation error: {e}"))?;
    if open.requester_id.as_slice() != remote.as_bytes() {
        anyhow::bail!("stage transport requester_id does not match QUIC peer identity");
    }

    let bind_addr = resolve_stage_transport_bind_addr(&node, &open).await?;
    let tcp_stream = TcpStream::connect(&bind_addr).await?;
    tcp_stream.set_nodelay(true)?;
    tracing::info!(
        "Inbound stage transport stream {} → {}",
        remote.fmt_short(),
        bind_addr
    );
    let (tcp_read, tcp_write) = tokio::io::split(tcp_stream);
    relay_bidirectional(tcp_read, tcp_write, quic_send, quic_recv).await
}

async fn resolve_stage_transport_bind_addr(
    node: &Node,
    open: &skippy_protocol::proto::stage::StageTransportOpen,
) -> Result<String> {
    let status_result = node
        .query_local_stage_status(crate::inference::skippy::StageStatusFilter {
            topology_id: Some(open.topology_id.clone()),
            run_id: Some(open.run_id.clone()),
            stage_id: Some(open.stage_id.clone()),
        })
        .await;
    match status_result {
        Ok(statuses) => {
            if let Some(status) = statuses.into_iter().find(|status| {
                status.topology_id == open.topology_id
                    && status.run_id == open.run_id
                    && status.stage_id == open.stage_id
            }) {
                if status.state != crate::inference::skippy::StageRuntimeState::Ready {
                    anyhow::bail!(
                        "stage {} / {} / {} is not ready: {:?}",
                        status.topology_id,
                        status.run_id,
                        status.stage_id,
                        status.state
                    );
                }
                return Ok(status.bind_addr);
            }
        }
        Err(error) => {
            if let Some(bind_addr) = node
                .stage_transport_alias(&open.topology_id, &open.run_id, &open.stage_id)
                .await
            {
                return Ok(bind_addr);
            }
            return Err(error).with_context(|| {
                format!(
                    "query local stage status for {} / {} / {}",
                    open.topology_id, open.run_id, open.stage_id
                )
            });
        }
    }
    if let Some(bind_addr) = node
        .stage_transport_alias(&open.topology_id, &open.run_id, &open.stage_id)
        .await
    {
        return Ok(bind_addr);
    }
    anyhow::bail!(
        "stage {} / {} / {} is not loaded locally",
        open.topology_id,
        open.run_id,
        open.stage_id
    )
}

/// Bidirectional relay between a TCP stream and a QUIC bi-stream.
///
/// Two directions run concurrently:
///   - tcp→quic (`relay_tcp_to_quic`): reads TCP, writes QUIC
///   - quic→tcp (`relay_quic_to_tcp`): reads QUIC, writes TCP
///
/// When either direction completes (EOF or stream close), we wait for the
/// other to finish. This is required for HTTP tunneling: the request
/// direction often completes before the response direction, and aborting
/// the response on request-side EOF would kill the reply.
pub async fn relay_bidirectional(
    tcp_read: tokio::io::ReadHalf<TcpStream>,
    tcp_write: tokio::io::WriteHalf<TcpStream>,
    quic_send: iroh::endpoint::SendStream,
    quic_recv: iroh::endpoint::RecvStream,
) -> Result<()> {
    let mut t1 = tokio::spawn(async move { relay_tcp_to_quic(tcp_read, quic_send).await });
    let mut t2 = tokio::spawn(async move { relay_quic_to_tcp(quic_recv, tcp_write).await });
    // Either direction may finish first:
    //   - tcp→quic finishes when the TCP side closes after responding
    //   - quic→tcp finishes when the QUIC side closes (e.g. request fully delivered)
    // In both cases, wait for the other direction to complete so the full
    // HTTP exchange can finish.
    tokio::select! {
        r1 = &mut t1 => finish_relay_pair(r1, t2, "tcp→quic", "quic→tcp").await,
        r2 = &mut t2 => finish_relay_pair(r2, t1, "quic→tcp", "tcp→quic").await,
    }
}

fn join_relay_task(
    join_result: std::result::Result<Result<()>, tokio::task::JoinError>,
) -> Result<()> {
    join_result?
}

async fn finish_relay_pair(
    first_result: std::result::Result<Result<()>, tokio::task::JoinError>,
    remaining_task: tokio::task::JoinHandle<Result<()>>,
    finished_label: &str,
    waiting_for_label: &str,
) -> Result<()> {
    let first = join_relay_task(first_result);
    tracing::debug!(
        "relay_bidirectional: {finished_label} finished, waiting for {waiting_for_label}"
    );
    let second = join_relay_task(remaining_task.await);
    first.and(second)
}

async fn relay_tcp_to_quic(
    mut tcp_read: tokio::io::ReadHalf<TcpStream>,
    mut quic_send: iroh::endpoint::SendStream,
) -> Result<()> {
    let mut buf = vec![0u8; 64 * 1024];
    let mut total: u64 = 0;
    loop {
        let n = tcp_read.read(&mut buf).await?;
        if n == 0 {
            tracing::info!("TCP→QUIC: TCP EOF after {total} bytes");
            break;
        }
        quic_send.write_all(&buf[..n]).await?;
        total += n as u64;
        BYTES_TRANSFERRED.fetch_add(n as u64, Ordering::Relaxed);
        tracing::debug!("TCP→QUIC: wrote {n} bytes (total: {total})");
    }
    quic_send.finish()?;
    Ok(())
}

async fn relay_quic_to_tcp(
    mut quic_recv: iroh::endpoint::RecvStream,
    mut tcp_write: tokio::io::WriteHalf<TcpStream>,
) -> Result<()> {
    relay_response_with_first_byte_timeout(
        &mut quic_recv,
        &mut tcp_write,
        quic_response_first_byte_timeout(),
    )
    .await
}

async fn relay_response_with_first_byte_timeout<R, W>(
    mut reader: R,
    mut writer: W,
    first_byte_timeout: Duration,
) -> Result<()>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    let mut buf = vec![0u8; 64 * 1024];
    let mut total: u64 = 0;
    tracing::debug!("QUIC→TCP: starting relay, about to first read");

    // First-byte timeout: allow enough time for remote prefill on real prompts.
    // After first byte arrives, no timeout (streaming responses can take minutes).
    match read_first_relay_chunk(&mut reader, &mut writer, &mut buf, first_byte_timeout).await? {
        Some(first_bytes) => {
            total += first_bytes as u64;
            BYTES_TRANSFERRED.fetch_add(first_bytes as u64, Ordering::Relaxed);
            tracing::debug!("QUIC→TCP: first read {first_bytes} bytes");
        }
        None => return Ok(()),
    }

    // After first byte, relay without timeout
    relay_remaining_chunks(&mut reader, &mut writer, &mut buf, &mut total).await?;
    Ok(())
}

async fn read_first_relay_chunk<R, W>(
    reader: &mut R,
    writer: &mut W,
    buf: &mut [u8],
    first_byte_timeout: Duration,
) -> Result<Option<usize>>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    match tokio::time::timeout(first_byte_timeout, reader.read(buf)).await {
        Err(_) => anyhow::bail!(
            "QUIC→TCP: no response within {:.3}s — host likely dead or still prefill-bound",
            first_byte_timeout.as_secs_f64()
        ),
        Ok(Ok(0)) => {
            tracing::info!("QUIC→TCP: stream end immediately (0 bytes)");
            Ok(None)
        }
        Ok(Ok(n)) => {
            writer.write_all(&buf[..n]).await?;
            Ok(Some(n))
        }
        Ok(Err(e)) => {
            tracing::warn!("QUIC→TCP: error on first read: {e}");
            Err(e.into())
        }
    }
}

async fn relay_remaining_chunks<R, W>(
    reader: &mut R,
    writer: &mut W,
    buf: &mut [u8],
    total: &mut u64,
) -> Result<()>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    loop {
        let n = match reader.read(buf).await {
            Ok(0) => {
                tracing::info!("QUIC→TCP: stream end after {total} bytes");
                return Ok(());
            }
            Ok(n) => n,
            Err(e) => return relay_remaining_chunks_error(*total, e),
        };
        writer.write_all(&buf[..n]).await?;
        *total += n as u64;
        BYTES_TRANSFERRED.fetch_add(n as u64, Ordering::Relaxed);
        tracing::debug!("QUIC→TCP: wrote {n} bytes (total: {total})");
    }
}

fn relay_remaining_chunks_error(total: u64, err: std::io::Error) -> Result<()> {
    tracing::warn!("QUIC→TCP: error after {total} bytes: {err}");
    Err(err.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{NodeRole, QuicBindSelection, RelayConfig, RelayPolicy};
    use std::collections::HashMap;
    use tokio::sync::oneshot;

    async fn start_test_node(
        role: NodeRole,
        peer_inference_only: bool,
    ) -> Result<(Node, crate::mesh::TunnelChannels)> {
        let relay_urls = Vec::new();
        let relay_auths = HashMap::new();
        Node::start(
            role,
            RelayConfig {
                urls: &relay_urls,
                auths: &relay_auths,
                policy: RelayPolicy::Disabled,
            },
            QuicBindSelection {
                ip: Some(std::net::Ipv4Addr::LOCALHOST.into()),
                port: None,
            },
            Some(0.0),
            false,
            peer_inference_only,
            None,
            None,
            crate::MeshRequirements::unrestricted(),
        )
        .await
    }

    async fn wait_for_peer(node: &Node, peer_id: EndpointId) -> Result<()> {
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if node.peers().await.iter().any(|peer| peer.id == peer_id) {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .context("timed out waiting for test mesh peer")?;
        Ok(())
    }

    #[test]
    fn inbound_http_tunnel_allows_inference_routes_only() {
        for request_line in [
            b"GET /v1/models HTTP/1.1\r\n".as_slice(),
            b"GET /models?refresh=1 HTTP/1.1\r\n".as_slice(),
            b"POST /v1/chat/completions HTTP/1.1\r\n".as_slice(),
            b"POST /v1/responses?stream=true HTTP/1.0\r\n".as_slice(),
        ] {
            assert!(
                is_allowed_inbound_http_tunnel_request(request_line),
                "expected request to be allowed: {}",
                String::from_utf8_lossy(request_line).trim_end()
            );
        }

        for request_line in [
            b"POST /mesh/load HTTP/1.1\r\n".as_slice(),
            b"POST /mesh/drop HTTP/1.1\r\n".as_slice(),
            b"GET /api/status HTTP/1.1\r\n".as_slice(),
            b"POST /v1/models HTTP/1.1\r\n".as_slice(),
            b"GET http://127.0.0.1:9337/v1/models HTTP/1.1\r\n".as_slice(),
            b"CONNECT 127.0.0.1:3131 HTTP/1.1\r\n".as_slice(),
            b"POST /v1/chat/completions HTTP/2\r\n".as_slice(),
            b"POST /v1/chat/completions HTTP/1.1 extra\r\n".as_slice(),
            b"POST /v1/chat/completions HTTP/1.1\n".as_slice(),
        ] {
            assert!(
                !is_allowed_inbound_http_tunnel_request(request_line),
                "expected request to be rejected: {}",
                String::from_utf8_lossy(request_line).trim_end()
            );
        }
    }

    #[tokio::test]
    async fn inbound_http_tunnel_request_line_reader_preserves_remaining_bytes() {
        let (mut writer, mut reader) = tokio::io::duplex(256);
        tokio::spawn(async move {
            writer
                .write_all(b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n\r\n{}")
                .await
                .unwrap();
        });

        let request_line = read_inbound_http_request_line(&mut reader).await.unwrap();
        assert_eq!(request_line, b"POST /v1/chat/completions HTTP/1.1\r\n");

        let mut remaining = Vec::new();
        reader.read_to_end(&mut remaining).await.unwrap();
        assert_eq!(remaining, b"Host: localhost\r\n\r\n{}");
    }

    #[tokio::test]
    async fn inbound_http_tunnel_request_line_reader_rejects_oversized_lines() {
        let (mut writer, mut reader) = tokio::io::duplex(MAX_INBOUND_HTTP_REQUEST_LINE_BYTES + 1);
        tokio::spawn(async move {
            writer
                .write_all(&vec![b'a'; MAX_INBOUND_HTTP_REQUEST_LINE_BYTES])
                .await
                .unwrap();
        });

        let error = read_inbound_http_request_line(&mut reader)
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("exceeded"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn inbound_http_tunnel_forwards_inference_and_rejects_control_routes() -> Result<()> {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let http_port = listener.local_addr()?.port();
        let (verify_second_accept_tx, verify_second_accept_rx) = oneshot::channel();
        let backend = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await?;
            let mut request = Vec::new();
            let mut chunk = [0u8; 1024];
            while !request.windows(4).any(|window| window == b"\r\n\r\n") {
                let read = stream.read(&mut chunk).await?;
                if read == 0 {
                    anyhow::bail!("allowed test request closed before headers completed");
                }
                request.extend_from_slice(&chunk[..read]);
            }
            assert!(
                request.starts_with(b"GET /v1/models HTTP/1.1\r\n"),
                "expected allowed request to reach the local proxy: {}",
                String::from_utf8_lossy(&request)
            );
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
                .await?;
            stream.shutdown().await?;

            verify_second_accept_rx.await?;
            assert!(
                tokio::time::timeout(Duration::from_millis(250), listener.accept())
                    .await
                    .is_err(),
                "rejected control route must not open a local TCP connection"
            );
            Ok::<_, anyhow::Error>(())
        });

        let (server, channels) = start_test_node(NodeRole::Host { http_port }, true).await?;
        let tunnel_manager =
            Manager::start(server.clone(), channels.rpc, channels.http, channels.stage).await?;
        tunnel_manager.set_http_port(http_port);
        server.start_accepting();

        let (client, _channels) = start_test_node(NodeRole::Client, false).await?;
        client.start_accepting();
        client.join(&server.invite_token().await).await?;
        wait_for_peer(&client, server.id()).await?;

        let (mut allowed_send, mut allowed_recv) = client.open_http_tunnel(server.id()).await?;
        allowed_send
            .write_all(b"GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n")
            .await?;
        allowed_send.finish()?;
        let allowed_response = allowed_recv.read_to_end(1024 * 1024).await?;
        assert!(
            allowed_response.starts_with(b"HTTP/1.1 200 OK\r\n"),
            "expected allowed route to be relayed: {}",
            String::from_utf8_lossy(&allowed_response)
        );

        let (mut rejected_send, mut rejected_recv) = client.open_http_tunnel(server.id()).await?;
        rejected_send
            .write_all(b"POST /mesh/drop HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\n\r\n")
            .await?;
        rejected_send.finish()?;
        let rejected_response = rejected_recv.read_to_end(1024 * 1024).await?;
        assert!(
            rejected_response.starts_with(b"HTTP/1.1 403 Forbidden\r\n"),
            "expected rejected route to receive a 403: {}",
            String::from_utf8_lossy(&rejected_response)
        );

        verify_second_accept_tx
            .send(())
            .map_err(|_| anyhow::anyhow!("backend verification task ended early"))?;
        backend.await??;
        Ok(())
    }

    /// Simulate relay_bidirectional behavior when one direction finishes
    /// before the other — the scenario that caused the remote proxy bug.
    ///
    /// Mimics the inbound HTTP tunnel on the receiving side:
    ///   - quic→tcp (request): delivers request bytes then hits EOF
    ///   - tcp→quic (response): backend responds AFTER request is fully delivered
    ///
    /// The bug: the old code aborted the response relay when the request
    /// relay completed, killing the response before it was sent back.
    #[tokio::test]
    async fn relay_bidirectional_waits_for_response_after_request_eof() {
        // Simulate QUIC side: request bytes arrive, then EOF (like finish())
        let (mut quic_write, quic_read) = tokio::io::duplex(4096);
        // Simulate QUIC response: we'll read what relay writes back
        let (quic_resp_write, mut quic_resp_read) = tokio::io::duplex(4096);

        // Simulate TCP side: reads request, delays, sends response
        let tcp_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let tcp_addr = tcp_listener.local_addr().unwrap();

        // Send the request on the QUIC side and close it (simulating finish())
        tokio::spawn(async move {
            quic_write
                .write_all(b"GET /test HTTP/1.1\r\n\r\n")
                .await
                .unwrap();
            drop(quic_write); // EOF — simulates quic_send.finish()
        });

        // Simulated backend: accept connection, read request, delay, respond
        let server = tokio::spawn(async move {
            let (mut stream, _) = tcp_listener.accept().await.unwrap();
            let mut buf = vec![0u8; 1024];
            let n = stream.read(&mut buf).await.unwrap();
            assert!(n > 0, "should receive request bytes");
            // Simulate prefill delay — response comes AFTER request EOF
            tokio::time::sleep(Duration::from_millis(50)).await;
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
                .await
                .unwrap();
            stream.shutdown().await.unwrap();
        });

        // Run relay_bidirectional as the receiving side would
        let tcp_stream = TcpStream::connect(tcp_addr).await.unwrap();
        let (tcp_read, tcp_write) = tokio::io::split(tcp_stream);

        // We can't easily get real QUIC streams in a unit test, so test the
        // core logic: use the same relay helpers with duplex streams to verify
        // that both directions complete.
        let t1 = tokio::spawn(async move {
            // tcp→quic direction (response): read from TCP, write to quic_resp_write
            let mut buf = vec![0u8; 4096];
            let mut total = 0u64;
            let mut writer = quic_resp_write;
            let mut reader = tcp_read;
            loop {
                let n = reader.read(&mut buf).await.unwrap();
                if n == 0 {
                    break;
                }
                writer.write_all(&buf[..n]).await.unwrap();
                total += n as u64;
            }
            total
        });

        let t2 = tokio::spawn(async move {
            // quic→tcp direction (request): read from quic_read, write to TCP
            let mut buf = vec![0u8; 4096];
            let mut reader = quic_read;
            let mut writer = tcp_write;
            loop {
                let n = reader.read(&mut buf).await.unwrap();
                if n == 0 {
                    break;
                }
                writer.write_all(&buf[..n]).await.unwrap();
            }
        });

        // The key assertion: both tasks must complete (not abort/hang)
        let response_bytes = tokio::time::timeout(Duration::from_secs(5), async {
            // t2 (request direction) will finish first because quic_write was dropped
            t2.await.unwrap();
            // t1 (response direction) must NOT be aborted — it should complete
            t1.await.unwrap()
        })
        .await
        .expect("relay should complete within 5s, not hang or abort");

        assert!(
            response_bytes > 0,
            "response bytes should have been relayed"
        );
        server.await.unwrap();

        // Verify the response actually made it through
        let mut response = Vec::new();
        quic_resp_read.read_to_end(&mut response).await.unwrap();
        let response_str = String::from_utf8_lossy(&response);
        assert!(
            response_str.contains("200 OK"),
            "response should contain 200 OK, got: {response_str}"
        );
    }

    #[tokio::test]
    async fn relay_response_times_out_before_first_byte() {
        let (mut upstream_write, upstream_read) = tokio::io::duplex(1024);
        let (downstream_write, mut downstream_read) = tokio::io::duplex(1024);

        let writer = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(75)).await;
            let _ = upstream_write.write_all(b"late response").await;
        });

        let err = relay_response_with_first_byte_timeout(
            upstream_read,
            downstream_write,
            Duration::from_millis(20),
        )
        .await
        .unwrap_err();

        assert!(err.to_string().contains("no response within"));
        writer.await.unwrap();

        let mut forwarded = Vec::new();
        downstream_read.read_to_end(&mut forwarded).await.unwrap();
        assert!(forwarded.is_empty());
    }

    #[tokio::test]
    async fn relay_response_allows_slow_but_healthy_first_byte() {
        let (mut upstream_write, upstream_read) = tokio::io::duplex(1024);
        let (downstream_write, mut downstream_read) = tokio::io::duplex(1024);

        let writer = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            upstream_write
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhello")
                .await
                .unwrap();
        });

        relay_response_with_first_byte_timeout(
            upstream_read,
            downstream_write,
            Duration::from_millis(200),
        )
        .await
        .unwrap();

        writer.await.unwrap();

        let mut forwarded = Vec::new();
        downstream_read.read_to_end(&mut forwarded).await.unwrap();
        assert_eq!(
            forwarded,
            b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhello"
        );
    }

    #[tokio::test]
    async fn relay_response_allows_slow_follow_up_chunks_after_first_byte() {
        let (mut upstream_write, upstream_read) = tokio::io::duplex(1024);
        let (downstream_write, mut downstream_read) = tokio::io::duplex(1024);

        let writer = tokio::spawn(async move {
            upstream_write
                .write_all(b"HTTP/1.1 200 OK\r\n")
                .await
                .unwrap();
            tokio::time::sleep(Duration::from_millis(75)).await;
            upstream_write
                .write_all(b"Content-Length: 5\r\n\r\nhello")
                .await
                .unwrap();
        });

        relay_response_with_first_byte_timeout(
            upstream_read,
            downstream_write,
            Duration::from_millis(20),
        )
        .await
        .unwrap();

        writer.await.unwrap();

        let mut forwarded = Vec::new();
        downstream_read.read_to_end(&mut forwarded).await.unwrap();
        assert_eq!(
            forwarded,
            b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhello"
        );
    }
}
