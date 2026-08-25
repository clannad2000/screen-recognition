class FrameSampler {
  /**
   * @param {HTMLVideoElement} videoElement
   * @param {HTMLCanvasElement} canvasElement  // 隐藏的离屏 canvas
   */
  constructor(videoElement, canvasElement) {
    this.video = videoElement;
    this.canvas = canvasElement;
    this.ctx = canvasElement.getContext('2d');
  }

  /**
   * 从当前视频帧以原始分辨率截取无损 PNG。
   * @returns {{ imageDataUrl: string, width: number, height: number, videoTimeMs: number }}
   */
  capture() {
    const video = this.video;
    const width = video.videoWidth;
    const height = video.videoHeight;
    if (!width || !height) {
      throw new Error('视频帧尚未就绪，无法截取。');
    }
    this.canvas.width = width;
    this.canvas.height = height;
    this.ctx.drawImage(video, 0, 0, this.canvas.width, this.canvas.height);
    return {
      imageDataUrl: this.canvas.toDataURL('image/png'),
      width:        this.canvas.width,
      height:       this.canvas.height,
      videoTimeMs:  Math.round(video.currentTime * 1000),
    };
  }
}
