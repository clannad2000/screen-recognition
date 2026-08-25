// 辅助函数
function formatVideoTime(ms) {
  const totalSeconds = Math.floor(ms / 1000);
  const h = Math.floor(totalSeconds / 3600);
  const m = Math.floor((totalSeconds % 3600) / 60);
  const s = totalSeconds % 60;
  const mm = String(m).padStart(2, '0');
  const ss = String(s).padStart(2, '0');
  return h > 0 ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
}

const STATE_COLORS = {
  '正常': '#16a34a',
  '蓝屏': '#2563eb',
  '黑屏': '#dc2626',
  '贴纸': '#d97706',
};
function stateColor(state) {
  return STATE_COLORS[state] || '#6b7280';
}

class VideoRecognitionController {
  constructor({ video, overlayCanvas, captureCanvas, config, uiRefs }) {
    this.video         = video;
    this.overlayCanvas = overlayCanvas;
    this.overlayCtx    = overlayCanvas.getContext('2d');
    this.config = Object.assign({
      sampleIntervalMs:     3000,
      confirmCount:         3,
      minConfidence:        0.70,
      trackingIouThreshold: 0.50,
      maxMissingFrames:     3,
    }, config);

    this.ui = uiRefs; // { statusEl, roundsEl, tracksEl, eventsEl, aiStatusEl }

    this.frameSampler = new FrameSampler(video, captureCanvas);
    this.tracker = new ScreenTracker({
      iouThreshold:     this.config.trackingIouThreshold,
      maxMissingFrames: this.config.maxMissingFrames,
    });
    this.stateMachine = new ScreenStateMachine({
      confirmCount:  this.config.confirmCount,
      minConfidence: this.config.minConfidence,
    });

    this.running               = false;
    this.inferenceRunning      = false;
    this.lastSampleVideoTimeMs = null;
    this.events                = [];
    this.rounds                = [];
    this.roundNumber           = 0;
    this.lastOverlayTracks     = [];
    this.consecutiveErrors     = 0;
    this.sessionId             = 0;
    this.activeAbortController = null;
    this._rafId                = null;

    // seek 事件
    this._onSeeked = () => {
      if (!this.running) return;
      this.tracker.reset();
      this.lastSampleVideoTimeMs = null;
      this._setAiStatus('视频位置已跳转，已重新建立屏幕跟踪');
      this._clearOverlay();
    };
    this._onEnded = () => {
      this.stop();
      this._setAiStatus('视频播放结束');
    };
    video.addEventListener('seeked', this._onSeeked);
    video.addEventListener('ended', this._onEnded);
  }

  updateConfig(config) {
    Object.assign(this.config, config);
    this.tracker.iouThreshold     = this.config.trackingIouThreshold;
    this.tracker.maxMissingFrames = this.config.maxMissingFrames;
    this.stateMachine.confirmCount  = this.config.confirmCount;
    this.stateMachine.minConfidence = this.config.minConfidence;
  }

  start() {
    if (this.running) return;
    this.sessionId++;
    this.running               = true;
    this.inferenceRunning      = false;
    this.lastSampleVideoTimeMs = null;
    this.consecutiveErrors     = 0;
    this.rounds                = [];
    this.roundNumber           = 0;
    this.tracker.reset();
    this._renderRounds([]);
    this._setAiStatus('检测运行中');
    this._tick();
  }

  stop() {
    this.running = false;
    if (this._rafId) { cancelAnimationFrame(this._rafId); this._rafId = null; }
    this.activeAbortController?.abort();
    this.activeAbortController = null;
    this.inferenceRunning = false;
    this._setAiStatus('检测已停止');
  }

  reset() {
    this.stop();
    this.tracker.reset();
    this.events = [];
    this.rounds = [];
    this.roundNumber = 0;
    this.lastSampleVideoTimeMs = null;
    this.consecutiveErrors = 0;
    this._clearOverlay();
    this._renderTracks([]);
    this._renderRounds([]);
    this._renderEvents([]);
    this._setAiStatus('已重置');
  }

  destroy() {
    this.stop();
    this.video.removeEventListener('seeked', this._onSeeked);
    this.video.removeEventListener('ended', this._onEnded);
  }

  _tick() {
    if (!this.running) return;
    this._rafId = requestAnimationFrame(() => this._tick());

    const currentMs = Math.round(this.video.currentTime * 1000);
    if (
      !this.video.paused     &&
      !this.video.ended      &&
      !this.inferenceRunning &&
      (
        this.lastSampleVideoTimeMs === null ||
        currentMs - this.lastSampleVideoTimeMs >= this.config.sampleIntervalMs
      )
    ) {
      this._processFrame(currentMs);
    }
  }

  async _processFrame(currentMs) {
    if (!this.running || this.inferenceRunning) return;
    this.inferenceRunning      = true;
    this.lastSampleVideoTimeMs = currentMs;
    const capturedSession      = this.sessionId;
    const roundNumber          = ++this.roundNumber;
    const recognitionStartedAt = performance.now();

    const abortController = new AbortController();
    this.activeAbortController = abortController;

    try {
      this._setAiStatus(`第 ${roundNumber} 轮识别中…`);
      const frame  = this.frameSampler.capture();
      const result = await RecognitionApi.recognize(frame.imageDataUrl, abortController.signal);

      // session 检查：防止旧请求污染新 session
      if (capturedSession !== this.sessionId) return;

      this.consecutiveErrors = 0;
      const elapsedMs = performance.now() - recognitionStartedAt;

      const observations = result.detections.map(d => ({
        bbox:              d.bbox_2d,
        state:             d.state,
        confidence:        d.confidence,
        sourceDetectionId: d.id,
        videoTimeMs:       currentMs,
        observedAt:        Date.now(),
      }));

      const trackingResult = this.tracker.update(observations);

      for (const { track, observation } of trackingResult.observedTracks) {
        const transition = this.stateMachine.applyObservation(track, observation);
        if (transition.event) {
          this.events.push(transition.event);
        }
      }

      this._recordRound({
        number: roundNumber,
        videoTimeMs: currentMs,
        elapsedMs,
        detectionCount: result.detections.length,
      });

      // 单次确认模式下直接使用本轮模型结果绘制，确保响应返回后立即标框。
      const overlayTracks = this.config.confirmCount === 1
        ? trackingResult.observedTracks.map(({ track, observation }) => ({
            ...track,
            bbox: observation.bbox,
            confirmedState: observation.state,
            latestState: observation.state,
            latestConfidence: observation.confidence,
            candidateState: null,
            candidateCount: 0,
          }))
        : this.tracker.getTracks();
      this._renderOverlay(overlayTracks);
      this._renderTracks(this.tracker.getTracks());
      this._renderEvents(this.events);
      this._setAiStatus(`第 ${roundNumber} 轮完成，用时 ${(elapsedMs / 1000).toFixed(2)} 秒；等待下一帧（${formatVideoTime(currentMs)}）`);

    } catch (err) {
      if (capturedSession !== this.sessionId) return;
      if (err.name === 'AbortError') return;
      const elapsedMs = performance.now() - recognitionStartedAt;
      this._recordRound({
        number: roundNumber,
        videoTimeMs: currentMs,
        elapsedMs,
        error: err.message || String(err),
      });
      this.consecutiveErrors++;
      const msg = this.consecutiveErrors >= 3
        ? `模型连续 ${this.consecutiveErrors} 次请求失败：${err.message}`
        : `识别出错：${err.message}`;
      this._setAiStatus(msg, true);
    } finally {
      if (capturedSession === this.sessionId) {
        this.inferenceRunning = false;
      }
    }
  }

  // ─── Renderer ───────────────────────────────────────────────────────────────

  _clearOverlay() {
    this.lastOverlayTracks = [];
    this.overlayCtx.clearRect(0, 0, this.overlayCanvas.width, this.overlayCanvas.height);
  }

  refreshOverlay() {
    this._renderOverlay(this.lastOverlayTracks);
  }

  _syncOverlaySize() {
    const w = this.video.clientWidth;
    const h = this.video.clientHeight;
    if (this.overlayCanvas.width !== w || this.overlayCanvas.height !== h) {
      this.overlayCanvas.width  = w;
      this.overlayCanvas.height = h;
    }
  }

  _renderOverlay(tracks) {
    this.lastOverlayTracks = tracks;
    this._syncOverlaySize();
    const ctx = this.overlayCtx;
    const W   = this.overlayCanvas.width;
    const H   = this.overlayCanvas.height;
    ctx.clearRect(0, 0, W, H);

    const activeTracks = tracks.filter(t => t.status === 'ACTIVE');
    for (const track of activeTracks) {
      const [x1, y1, x2, y2] = track.bbox;
      const left   = x1 / 1000 * W;
      const top    = y1 / 1000 * H;
      const right  = x2 / 1000 * W;
      const bottom = y2 / 1000 * H;
      const color  = stateColor(track.confirmedState ?? track.latestState);
      const lw     = Math.max(2, Math.round(W / 600));

      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth   = lw;
      ctx.strokeRect(left, top, right - left, bottom - top);
      ctx.restore();

      // 标签文本
      let label = `${track.id} | ${track.confirmedState ?? '初始化中'}`;
      if (track.candidateState && track.candidateState !== track.confirmedState) {
        label += ` → ${track.candidateState} ${track.candidateCount}/${this.config.confirmCount}`;
      }
      this._drawOverlayLabel(label, left, top, color, W);
    }
  }

  _drawOverlayLabel(text, x, y, color, canvasWidth) {
    const ctx      = this.overlayCtx;
    const fontSize = Math.max(13, Math.round(canvasWidth / 80));
    const padX     = 6;
    const padY     = 4;
    ctx.save();
    ctx.font         = `700 ${fontSize}px "Microsoft YaHei", "PingFang SC", sans-serif`;
    ctx.textBaseline = 'middle';
    const textW      = ctx.measureText(text).width;
    const boxW       = textW + padX * 2;
    const boxH       = fontSize + padY * 2;
    const boxX       = Math.max(0, Math.min(canvasWidth - boxW, x));
    const boxY       = y - boxH < 0 ? y : y - boxH;
    ctx.fillStyle    = color;
    ctx.globalAlpha  = 0.88;
    ctx.fillRect(boxX, boxY, boxW, boxH);
    ctx.globalAlpha  = 1;
    ctx.fillStyle    = '#fff';
    ctx.fillText(text, boxX + padX, boxY + boxH / 2);
    ctx.restore();
  }

  _renderTracks(tracks) {
    const el = this.ui.tracksEl;
    if (!el) return;
    el.innerHTML = '';
    const active = tracks.filter(t => t.status === 'ACTIVE');
    if (!active.length) {
      el.innerHTML = '<div class="result-empty">暂无检测到的屏幕。</div>';
      return;
    }
    for (const t of active) {
      const color = stateColor(t.confirmedState ?? t.latestState);
      const div = document.createElement('div');
      div.className = 'result-item';
      div.style.setProperty('--item-color', color);
      div.innerHTML = `
        <span class="result-id">${t.id}</span>
        <span class="result-state">${t.confirmedState ?? '初始化中'}</span>
        <span class="result-box">
          已确认：${t.confirmedState ?? '—'} &nbsp;|&nbsp;
          最新：${t.latestState ?? '—'} (${(t.latestConfidence * 100).toFixed(0)}%)
          ${t.candidateState ? `&nbsp;|&nbsp; 候选：${t.candidateState} ${t.candidateCount}/${this.config.confirmCount}` : ''}
        </span>`;
      el.appendChild(div);
    }
  }

  _recordRound(round) {
    this.rounds.push(round);
    if (this.rounds.length > 50) this.rounds.shift();
    this._renderRounds(this.rounds);
  }

  _renderRounds(rounds) {
    const el = this.ui.roundsEl;
    if (!el) return;
    el.replaceChildren();
    if (!rounds.length) {
      const empty = document.createElement('div');
      empty.className = 'result-empty';
      empty.textContent = '开始检测后将在这里显示每轮识别耗时。';
      el.appendChild(empty);
      return;
    }
    for (const round of [...rounds].reverse()) {
      const row = document.createElement('div');
      row.className = 'result-item';
      row.style.setProperty('--item-color', round.error ? '#dc2626' : '#2563eb');
      const title = document.createElement('span');
      title.className = 'result-id';
      title.textContent = `第 ${round.number} 轮 · ${formatVideoTime(round.videoTimeMs)}`;
      const result = document.createElement('span');
      result.className = 'result-state';
      result.textContent = round.error ? '识别失败' : `检测到 ${round.detectionCount} 个屏幕`;
      const detail = document.createElement('span');
      detail.className = 'result-box';
      detail.textContent = `识别耗时：${(round.elapsedMs / 1000).toFixed(2)} 秒${round.error ? ` · ${round.error}` : ''}`;
      row.append(title, result, detail);
      el.appendChild(row);
    }
  }

  _renderEvents(events) {
    const el = this.ui.eventsEl;
    if (!el) return;
    el.innerHTML = '';
    if (!events.length) {
      el.innerHTML = '<div class="result-empty">暂无状态变化事件。</div>';
      return;
    }
    // 最新在上
    const reversed = [...events].reverse();
    for (const e of reversed) {
      const color = stateColor(e.toState);
      const div = document.createElement('div');
      div.className = 'result-item';
      div.style.setProperty('--item-color', color);
      div.innerHTML = `
        <span class="result-id">${formatVideoTime(e.videoTimeMs)} · ${e.trackId}</span>
        <span class="result-state">${e.fromState} → ${e.toState}</span>
        <span class="result-box">置信度 ${(e.confidence * 100).toFixed(0)}%</span>`;
      el.appendChild(div);
    }
  }

  _setAiStatus(msg, isError = false) {
    const el = this.ui.aiStatusEl;
    if (el) {
      el.textContent = msg;
      el.style.color = isError ? '#dc2626' : '';
    }

    const statusPanel = this.ui.statusEl;
    if (statusPanel) {
      statusPanel.textContent = msg;
      statusPanel.className   = 'status' + (isError ? ' error' : '');
    }
  }
}
