/**
 * 封装 /api/recognize HTTP 调用，图片模式和视频模式共用。
 */
class RecognitionApi {
  /** @returns {Promise<{selected: string, models: Array}>} */
  static async models() {
    const response = await fetch('/api/models');
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    if (!Array.isArray(data.models)) throw new Error('模型列表格式无效');
    return data;
  }

  /**
   * @param {string} imageDataUrl
   * @param {string} modelProfile
   * @param {AbortSignal} [signal]
   * @returns {Promise<{detections: Array, meta: Object}>}
   */
  static async recognize(imageDataUrl, modelProfile, signal) {
    const response = await fetch('/api/recognize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: imageDataUrl, modelProfile }),
      signal,
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    if (!Array.isArray(data.detections)) throw new Error('detections 格式无效');
    return data;
  }
}
