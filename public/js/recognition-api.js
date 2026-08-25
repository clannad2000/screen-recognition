/**
 * 封装 /api/recognize HTTP 调用，图片模式和视频模式共用。
 */
class RecognitionApi {
  /**
   * @param {string} imageDataUrl
   * @param {AbortSignal} [signal]
   * @returns {Promise<{detections: Array, meta: Object}>}
   */
  static async recognize(imageDataUrl, signal) {
    const response = await fetch('/api/recognize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: imageDataUrl }),
      signal,
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    if (!Array.isArray(data.detections)) throw new Error('detections 格式无效');
    return data;
  }
}
