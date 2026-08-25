/**
 * @typedef {Object} StateChangeEvent
 * @property {string}   id
 * @property {string}   trackId
 * @property {string|null} fromState
 * @property {string}   toState
 * @property {number}   videoTimeMs
 * @property {number}   occurredAt
 * @property {number}   confidence
 * @property {number[]} bbox
 */

class ScreenStateMachine {
  constructor(options = {}) {
    this.confirmCount  = options.confirmCount  ?? 3;
    this.minConfidence = options.minConfidence ?? 0.70;
    this._eventSeq     = 0;
  }

  /**
   * @param {ScreenTrack} track
   * @param {RecognitionObservation} observation
   * @returns {{ changed: boolean, initialized?: boolean, event: StateChangeEvent|null }}
   */
  applyObservation(track, observation) {
    // 始终更新 latestState/latestConfidence
    track.latestState      = observation.state;
    track.latestConfidence = observation.confidence;

    // 置信度不足：跳过状态推进，但不重置候选
    if (observation.confidence < this.minConfidence) {
      return { changed: false, event: null };
    }

    const state = observation.state;

    // observation == confirmedState：清除候选
    if (track.confirmedState === state) {
      track.candidateState = null;
      track.candidateCount = 0;
      return { changed: false, event: null };
    }

    // 推进候选计数
    if (track.candidateState === state) {
      track.candidateCount++;
    } else {
      track.candidateState = state;
      track.candidateCount = 1;
    }

    // 未达到 confirmCount
    if (track.candidateCount < this.confirmCount) {
      return { changed: false, event: null };
    }

    // 达到 confirmCount，切换状态
    const oldState = track.confirmedState;
    track.confirmedState = state;
    track.candidateState = null;
    track.candidateCount = 0;

    // 初始确认（null → state），不产生事件
    if (oldState === null) {
      return { changed: false, initialized: true, event: null };
    }

    // 真正的状态变化
    const event = {
      id:          `event-${++this._eventSeq}`,
      trackId:     track.id,
      fromState:   oldState,
      toState:     state,
      videoTimeMs: observation.videoTimeMs,
      occurredAt:  observation.observedAt,
      confidence:  observation.confidence,
      bbox:        [...track.bbox],
    };
    return { changed: true, event };
  }

  resetTrack(track) {
    track.confirmedState = null;
    track.candidateState = null;
    track.candidateCount = 0;
  }
}
