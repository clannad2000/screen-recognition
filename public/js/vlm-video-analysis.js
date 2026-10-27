class VlmVideoApi {
  static async create(file, config, signal) {
    const query = new URLSearchParams({
      modelProfile: config.modelProfile,
      sampleInterval: String(config.sampleInterval),
      windowSize: String(config.windowSize),
      windowStep: String(config.windowStep),
      startSec: String(config.startSec),
      endSec: String(config.endSec),
    });
    const response = await fetch(`/api/vlm-video/jobs?${query}`, {
      method: 'POST',
      headers: {
        'Content-Type': file.type || 'application/octet-stream',
        'X-File-Name': encodeURIComponent(file.name),
      },
      body: file,
      signal,
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    if (!data.jobId) throw new Error('服务端未返回视频分析任务编号。');
    return data;
  }

  static async status(jobId, after = 0) {
    const response = await fetch(
      `/api/vlm-video/jobs/${encodeURIComponent(jobId)}?after=${encodeURIComponent(after)}`,
      { cache: 'no-store' },
    );
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    return data;
  }

  static async reset(jobId) {
    const response = await fetch(
      `/api/vlm-video/jobs/${encodeURIComponent(jobId)}/reset`,
      { method: 'POST' },
    );
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    return data;
  }
}

class VlmVideoAnalysisController {
  static STORAGE_KEY = 'screen-recognition.vlm-video-job-id';

  constructor({ ui, getModelProfile, canStart, onRunningChange }) {
    this.ui = ui;
    this.getModelProfile = getModelProfile;
    this.canStart = canStart;
    this.onRunningChange = onRunningChange;
    this.file = null;
    this.objectUrl = null;
    this.jobId = null;
    this.running = false;
    this.lastSequence = 0;
    this.renderedSequences = new Set();
    this.pollTimer = null;
    this.uploadAbortController = null;
    this.sessionId = 0;

    this.ui.video.addEventListener('timeupdate', () => this._updateVideoTime());
  }

  loadFile(file) {
    if (!file || this.running || this.jobId) return;
    if (file.size > 4 * 1024 * 1024 * 1024) {
      this._setStatus('视频文件不能超过 4 GB。', 'error');
      return;
    }
    const allowed = /\.(mp4|mov|webm|avi|mkv|m4v)$/i;
    if (!allowed.test(file.name)) {
      this._setStatus('仅支持 MP4、MOV、WebM、AVI、MKV 或 M4V 视频。', 'error');
      return;
    }

    this._revokeObjectUrl();
    this.file = file;
    this.objectUrl = URL.createObjectURL(file);
    this.ui.video.src = this.objectUrl;
    this.ui.startSec.value = '0';
    this.ui.endSec.value = '';
    this.ui.startSec.removeAttribute('max');
    this.ui.endSec.removeAttribute('max');
    this.ui.uploadTitle.textContent = file.name;
    this.ui.video.onerror = () => {
      this._setStatus(
        '浏览器无法预览该编码格式；若后端 OpenCV 可以解码，仍可启动分析。',
        'error',
      );
      this.ui.headerBadge.textContent = '预览不可用';
      this._setControls();
    };
    this.ui.video.onloadedmetadata = () => {
      const duration = Number(this.ui.video.duration);
      this.ui.videoStage.style.display = '';
      this.ui.videoEmpty.style.display = 'none';
      this.ui.startSec.value = '0';
      if (Number.isFinite(duration)) {
        const value = duration.toFixed(3).replace(/\.000$/, '');
        this.ui.endSec.value = value;
        this.ui.startSec.max = value;
        this.ui.endSec.max = value;
      }
      this.ui.resolutionBadge.textContent =
        `分辨率：${this.ui.video.videoWidth} × ${this.ui.video.videoHeight}`;
      this.ui.headerBadge.textContent = '视频已加载';
      this._updateVideoTime();
      this._setStatus('视频加载成功，确认时间范围后开始分析。', 'ok');
      this._setControls();
    };
    this._clearResults();
    this._setControls();
  }

  async start() {
    if (!this.file || this.running || this.jobId) return;
    let config;
    try {
      config = this._readConfig();
    } catch (error) {
      this._setStatus(error.message, 'error');
      return;
    }

    const capturedSession = ++this.sessionId;
    this.running = true;
    this.lastSequence = 0;
    this.renderedSequences.clear();
    this._clearResults();
    this._notifyRunningChange();
    this._setControls();
    this._setStatus('正在上传视频并创建后台分析任务…');
    this.ui.headerBadge.textContent = '正在上传';
    this.ui.aiBadge.textContent = 'AI：准备中';
    this.uploadAbortController = new AbortController();
    this.ui.video.play().catch(() => {});

    try {
      const snapshot = await VlmVideoApi.create(
        this.file,
        config,
        this.uploadAbortController.signal,
      );
      if (capturedSession !== this.sessionId) return;
      this.jobId = snapshot.jobId;
      sessionStorage.setItem(VlmVideoAnalysisController.STORAGE_KEY, this.jobId);
      this.ui.headerBadge.textContent = '后台分析中';
      this._applySnapshot(snapshot);
      this._schedulePoll(capturedSession, 200);
    } catch (error) {
      if (capturedSession !== this.sessionId || error.name === 'AbortError') return;
      this.running = false;
      this._notifyRunningChange();
      this._setControls();
      this.ui.headerBadge.textContent = '启动失败';
      this.ui.aiBadge.textContent = 'AI：启动失败';
      this._setStatus(error.message || String(error), 'error');
    } finally {
      if (capturedSession === this.sessionId) this.uploadAbortController = null;
    }
  }

  async reset() {
    const jobId = this.jobId;
    const capturedSession = ++this.sessionId;
    this._clearPoll();
    this.uploadAbortController?.abort();
    this.uploadAbortController = null;
    this.running = Boolean(jobId);
    this._notifyRunningChange();
    this._setControls();
    this.ui.video.pause();
    this._setStatus(jobId ? '正在终止后台分析任务…' : '正在重置…');
    this.ui.headerBadge.textContent = '正在重置';

    if (jobId) {
      try {
        await VlmVideoApi.reset(jobId);
      } catch (error) {
        if (capturedSession !== this.sessionId) return;
        this.running = true;
        this._notifyRunningChange();
        this._setControls();
        this.ui.headerBadge.textContent = '重置失败';
        this._setStatus(`未能终止后台任务：${error.message || error}`, 'error');
        this._schedulePoll(capturedSession, 1500);
        return;
      }
    }

    if (capturedSession !== this.sessionId) return;
    this.jobId = null;
    this.running = false;
    this.lastSequence = 0;
    this.renderedSequences.clear();
    sessionStorage.removeItem(VlmVideoAnalysisController.STORAGE_KEY);
    this._clearResults();
    if (Number.isFinite(this.ui.video.duration)) this.ui.video.currentTime = 0;
    this._updateVideoTime();
    this.ui.headerBadge.textContent = this.file ? '已重置' : '未加载';
    this.ui.aiBadge.textContent = 'AI：待机';
    this._setStatus(
      this.file ? '已终止并清空分析结果，可以重新开始。' : '已重置。',
      'ok',
    );
    this._notifyRunningChange();
    this._setControls();
  }

  async restore() {
    const storedJobId = sessionStorage.getItem(VlmVideoAnalysisController.STORAGE_KEY);
    if (!storedJobId) return;
    const capturedSession = ++this.sessionId;
    try {
      const snapshot = await VlmVideoApi.status(storedJobId, 0);
      if (capturedSession !== this.sessionId) return;
      this.jobId = storedJobId;
      this.running = snapshot.status === 'running';
      this.ui.uploadTitle.textContent = snapshot.filename || '已恢复的分析任务';
      this.ui.headerBadge.textContent = this.running ? '已恢复后台任务' : '已恢复结果';
      this._applySnapshot(snapshot);
      this._notifyRunningChange();
      this._setControls();
      if (this.running) this._schedulePoll(capturedSession, 1000);
    } catch (_) {
      sessionStorage.removeItem(VlmVideoAnalysisController.STORAGE_KEY);
    }
  }

  destroy() {
    this._clearPoll();
    this._revokeObjectUrl();
  }

  refreshControls() {
    this._setControls();
  }

  _readConfig() {
    const sampleInterval = Number(this.ui.sampleInterval.value);
    const windowSize = Number(this.ui.windowSize.value);
    const windowStep = Number(this.ui.windowStep.value);
    const startSec = Number(this.ui.startSec.value);
    const endText = this.ui.endSec.value.trim();
    const endSec = endText === '' ? '' : Number(endText);
    if (!Number.isFinite(sampleInterval) || sampleInterval < 0.1) {
      throw new Error('抽帧间隔必须大于或等于 0.1 秒。');
    }
    if (!Number.isInteger(windowSize) || windowSize < 1 || windowSize > 20) {
      throw new Error('窗口大小必须是 1～20 的整数。');
    }
    if (!Number.isInteger(windowStep) || windowStep < 1) {
      throw new Error('窗口步长必须是正整数。');
    }
    if (!Number.isFinite(startSec) || startSec < 0) {
      throw new Error('开始时间不能小于 0。');
    }
    if (endSec !== '' && (!Number.isFinite(endSec) || endSec < startSec)) {
      throw new Error('结束时间必须大于或等于开始时间。');
    }
    return {
      modelProfile: this.getModelProfile(),
      sampleInterval,
      windowSize,
      windowStep,
      startSec,
      endSec,
    };
  }

  _schedulePoll(capturedSession, delay) {
    this._clearPoll();
    this.pollTimer = setTimeout(() => this._poll(capturedSession), delay);
  }

  async _poll(capturedSession) {
    if (capturedSession !== this.sessionId || !this.jobId) return;
    try {
      const snapshot = await VlmVideoApi.status(this.jobId, this.lastSequence);
      if (capturedSession !== this.sessionId) return;
      this._applySnapshot(snapshot);
      if (snapshot.status === 'running') {
        this._schedulePoll(capturedSession, 1000);
      }
    } catch (error) {
      if (capturedSession !== this.sessionId) return;
      this._setStatus(`读取分析进度失败，稍后自动重试：${error.message || error}`, 'error');
      this._schedulePoll(capturedSession, 2000);
    }
  }

  _applySnapshot(snapshot) {
    for (const result of snapshot.results || []) {
      const sequence = Number(result.sequence) || 0;
      if (this.renderedSequences.has(sequence)) continue;
      this.renderedSequences.add(sequence);
      this.lastSequence = Math.max(this.lastSequence, sequence);
      this._prependResult(result);
    }

    const sampled = Number(snapshot.sampledCount) || 0;
    const windows = Number(snapshot.sentWindowCount) || 0;
    this.ui.progressBadge.textContent = `抽帧：${sampled} · 结果：${windows}`;
    if (snapshot.status === 'running') {
      this.running = true;
      this.ui.aiBadge.textContent = `AI：后台分析中（已完成 ${windows} 轮）`;
      this._setStatus(`后台分析中：已抽取 ${sampled} 帧，已完成 ${windows} 个窗口。`);
    } else if (snapshot.status === 'completed') {
      this.running = false;
      this.ui.headerBadge.textContent = '分析完成';
      this.ui.aiBadge.textContent = 'AI：分析完成';
      const message = windows
        ? `分析完成，共生成 ${windows} 个窗口结果。`
        : `分析完成，但采样帧数量不足以组成大小为 ${this.ui.windowSize.value} 的窗口。`;
      this._setStatus(message, 'ok');
    } else if (snapshot.status === 'failed') {
      this.running = false;
      this.ui.headerBadge.textContent = '分析失败';
      this.ui.aiBadge.textContent = 'AI：分析失败';
      this._setStatus(snapshot.error || '后台分析任务失败。', 'error');
    }
    this._notifyRunningChange();
    this._setControls();
  }

  _prependResult(result) {
    if (this.ui.results.querySelector('.result-empty')) this.ui.results.replaceChildren();
    const card = document.createElement('article');
    const riskClass = result.suspicious === true && ['low', 'medium', 'high'].includes(result.riskLevel)
      ? ` vlm-risk-${result.riskLevel}`
      : '';
    card.className = `vlm-result-card${result.error ? ' error' : ''}${riskClass}`;

    const header = document.createElement('div');
    header.className = 'vlm-result-header';
    const title = document.createElement('strong');
    title.textContent = `第 ${result.windowNumber} 轮 · ${formatVideoTime(result.startVideoTimeMs)}–${formatVideoTime(result.endVideoTimeMs)}`;
    const elapsed = document.createElement('span');
    elapsed.className = 'badge';
    elapsed.textContent = `耗时：${Number(result.elapsedSeconds || 0).toFixed(2)} 秒`;
    header.append(title, elapsed);

    const summary = document.createElement('p');
    summary.className = 'vlm-result-summary';
    summary.textContent = result.error ? `本轮处理失败：${result.error}` : result.summary;

    if (!result.error) {
      const details = document.createElement('div');
      details.className = 'vlm-result-details';
      const riskLabels = { low: '低', medium: '中', high: '高' };
      const confidence = Number(result.confidence);
      const values = [
        ['是否可疑', result.suspicious === true ? '是' : result.suspicious === false ? '否' : '—'],
        ['风险等级', riskLabels[result.riskLevel] || '—'],
        ['置信度', Number.isFinite(confidence) ? `${(confidence * 100).toFixed(0)}%` : '—'],
      ];
      for (const [label, value] of values) {
        const detail = document.createElement('span');
        detail.className = `vlm-result-detail${label === '风险等级' && riskClass ? ' vlm-risk-detail' : ''}`;
        detail.textContent = `${label}：${value}`;
        details.appendChild(detail);
      }
      card.append(header, summary, details);
    } else {
      card.append(header, summary);
    }

    const gallery = document.createElement('div');
    gallery.className = 'vlm-frame-grid';
    for (const frame of result.images || []) {
      const figure = document.createElement('figure');
      const image = document.createElement('img');
      image.src = frame.url;
      image.alt = `${frame.frameId}，视频时间 ${formatVideoTime(frame.videoTimeMs)}`;
      image.loading = 'lazy';
      const caption = document.createElement('figcaption');
      caption.textContent = `${frame.frameId} · ${formatVideoTime(frame.videoTimeMs)} (${frame.videoTimeMs}ms)`;
      figure.append(image, caption);
      gallery.appendChild(figure);
    }
    card.append(gallery);
    this.ui.results.prepend(card);
  }

  _clearResults() {
    this.ui.results.replaceChildren();
    const empty = document.createElement('div');
    empty.className = 'result-empty';
    empty.textContent = '每个滑动窗口完成后，将在这里显示分析结论、风险信息和当次输入图片。';
    this.ui.results.appendChild(empty);
    this.ui.progressBadge.textContent = '抽帧：0 · 结果：0';
  }

  _setControls() {
    const locked = this.running || Boolean(this.jobId);
    this.ui.fileInput.disabled = locked;
    for (const input of this.ui.configInputs) input.disabled = locked;
    this.ui.startButton.disabled = !this.file || locked || !this.canStart();
    this.ui.resetButton.disabled = !this.file && !this.jobId && !this.running;
  }

  _setStatus(message, type = '') {
    this.ui.status.textContent = message;
    this.ui.status.className = `status${type ? ` ${type}` : ''}`;
  }

  _updateVideoTime() {
    const current = Number.isFinite(this.ui.video.currentTime) ? this.ui.video.currentTime : 0;
    const duration = Number.isFinite(this.ui.video.duration) ? this.ui.video.duration : 0;
    this.ui.timeBadge.textContent =
      `时间：${formatVideoTime(Math.round(current * 1000))} / ${duration ? formatVideoTime(Math.round(duration * 1000)) : '—'}`;
  }

  _notifyRunningChange() {
    this.onRunningChange?.(this.running);
  }

  _clearPoll() {
    if (this.pollTimer) clearTimeout(this.pollTimer);
    this.pollTimer = null;
  }

  _revokeObjectUrl() {
    if (this.objectUrl) URL.revokeObjectURL(this.objectUrl);
    this.objectUrl = null;
  }
}
