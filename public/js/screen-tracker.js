/**
 * @typedef {Object} RecognitionObservation
 * @property {number[]} bbox          [x1,y1,x2,y2] 0~1000
 * @property {string}   state
 * @property {number}   confidence
 * @property {string}   sourceDetectionId
 * @property {number}   videoTimeMs
 * @property {number}   observedAt
 */

/**
 * @typedef {Object} ScreenTrack
 * @property {string}  id
 * @property {number[]} bbox
 * @property {string|null} confirmedState
 * @property {string|null} candidateState
 * @property {number} candidateCount
 * @property {string|null} latestState
 * @property {number} latestConfidence
 * @property {number} firstSeenVideoTimeMs
 * @property {number} lastSeenVideoTimeMs
 * @property {number} lastObservedAt
 * @property {number} missingCount
 * @property {number} totalObservations
 * @property {"ACTIVE"|"LOST"} status
 */

function calculateIoU(a, b) {
  const left   = Math.max(a[0], b[0]);
  const top    = Math.max(a[1], b[1]);
  const right  = Math.min(a[2], b[2]);
  const bottom = Math.min(a[3], b[3]);
  const iw = Math.max(0, right - left);
  const ih = Math.max(0, bottom - top);
  const intersection = iw * ih;
  const areaA = Math.max(0, a[2]-a[0]) * Math.max(0, a[3]-a[1]);
  const areaB = Math.max(0, b[2]-b[0]) * Math.max(0, b[3]-b[1]);
  const union = areaA + areaB - intersection;
  return union > 0 ? intersection / union : 0;
}

function matchTracks(tracks, observations, threshold) {
  const candidates = [];
  for (let ti = 0; ti < tracks.length; ti++) {
    if (tracks[ti].status === 'LOST') continue;
    for (let oi = 0; oi < observations.length; oi++) {
      const iou = calculateIoU(tracks[ti].bbox, observations[oi].bbox);
      if (iou >= threshold) candidates.push({ trackIndex: ti, observationIndex: oi, iou });
    }
  }
  candidates.sort((a, b) => b.iou - a.iou);
  const usedTracks = new Set();
  const usedObs   = new Set();
  const matches   = [];
  for (const item of candidates) {
    if (usedTracks.has(item.trackIndex) || usedObs.has(item.observationIndex)) continue;
    usedTracks.add(item.trackIndex);
    usedObs.add(item.observationIndex);
    matches.push(item);
  }
  return { matches, usedTracks, usedObs };
}

class ScreenTracker {
  constructor(options = {}) {
    this.iouThreshold     = options.iouThreshold    ?? 0.50;
    this.maxMissingFrames = options.maxMissingFrames ?? 3;
    this._tracks  = [];
    this._nextId  = 1;
  }

  /**
   * @param {RecognitionObservation[]} observations
   * @returns {{ tracks: ScreenTrack[], observedTracks: {track:ScreenTrack, observation:RecognitionObservation}[], created: ScreenTrack[], lost: ScreenTrack[] }}
   */
  update(observations) {
    const { matches, usedTracks, usedObs } = matchTracks(
      this._tracks, observations, this.iouThreshold
    );

    const observedTracks = [];

    // 更新已匹配 track
    for (const m of matches) {
      const track = this._tracks[m.trackIndex];
      const obs   = observations[m.observationIndex];
      track.bbox                = obs.bbox;
      track.lastSeenVideoTimeMs = obs.videoTimeMs;
      track.lastObservedAt      = obs.observedAt;
      track.missingCount        = 0;
      track.totalObservations  += 1;
      track.status              = 'ACTIVE';
      observedTracks.push({ track, observation: obs });
    }

    // 未匹配的 track：missingCount++
    const lost = [];
    for (let ti = 0; ti < this._tracks.length; ti++) {
      if (this._tracks[ti].status === 'LOST') continue;
      if (!usedTracks.has(ti)) {
        this._tracks[ti].missingCount++;
        if (this._tracks[ti].missingCount >= this.maxMissingFrames) {
          this._tracks[ti].status = 'LOST';
          lost.push(this._tracks[ti]);
        }
      }
    }

    // 未匹配的 observation：创建新 track
    const created = [];
    for (let oi = 0; oi < observations.length; oi++) {
      if (usedObs.has(oi)) continue;
      const obs   = observations[oi];
      const track = {
        id:                   `track-${this._nextId++}`,
        bbox:                 obs.bbox,
        confirmedState:       null,
        candidateState:       null,
        candidateCount:       0,
        latestState:          obs.state,
        latestConfidence:     obs.confidence,
        firstSeenVideoTimeMs: obs.videoTimeMs,
        lastSeenVideoTimeMs:  obs.videoTimeMs,
        lastObservedAt:       obs.observedAt,
        missingCount:         0,
        totalObservations:    1,
        status:               'ACTIVE',
      };
      this._tracks.push(track);
      created.push(track);
      observedTracks.push({ track, observation: obs });
    }

    return { tracks: this._tracks, observedTracks, created, lost };
  }

  getTracks() { return this._tracks; }

  reset() {
    this._tracks = [];
    this._nextId = 1;
  }
}
