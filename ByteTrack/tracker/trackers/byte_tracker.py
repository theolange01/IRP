# Ultralytics YOLO 🚀, AGPL-3.0 license

import numpy as np
from scipy.spatial import distance

from ..utils import matching
from ..utils.kalman_filter import KalmanFilterXYAH
from .basetrack import BaseTrack, TrackState


class STrack(BaseTrack):
    shared_kalman = KalmanFilterXYAH()

    def __init__(self, tlwh, score, cls):
        """wait activate."""
        self._tlwh = np.asarray(self.tlbr_to_tlwh(tlwh[:-1]), dtype=np.float32)
        self.new_tlwh = self._tlwh
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0
        self.cls = cls
        self.idx = tlwh[-1]
        self.positions = [self._tlwh]

    def predict(self):
        """Predicts mean and covariance using Kalman filter."""
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        """Perform multi-object predictive tracking using Kalman filter for given stracks."""
        if len(stracks) <= 0:
            return
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][7] = 0
        multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        """Update state tracks positions and covariances using a homography matrix."""
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet."""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.convert_coords(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        """Reactivates a previously lost track with a new detection."""
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance,
                                                               self.convert_coords(new_track.tlwh))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.cls = new_track.cls
        self.idx = new_track.idx

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh

        self.positions.append(new_tlwh)

        self.new_tlwh = new_tlwh
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance,
                                                               self.convert_coords(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.cls = new_track.cls
        self.idx = new_track.idx

    def convert_coords(self, tlwh):
        """Convert a bounding box's top-left-width-height format to its x-y-angle-height equivalent."""
        return self.tlwh_to_xyah(tlwh)

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        """Converts top-left bottom-right format to top-left width height format."""
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        """Converts tlwh bounding box format to tlbr format."""
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        """Return a string representation of the BYTETracker object with start and end frames and track ID."""
        return f'OT_{self.track_id}_({self.start_frame}-{self.end_frame})'


class BYTETracker:

    def __init__(self, args, frame_rate=30):
        """Initialize a YOLOv8 object to track objects with given arguments and frame rate."""
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        self.filtered_stracks = []
        self.motion_history = []

        self.frame_id = 0
        self.args = args
        self.max_time_lost = int(frame_rate / 30.0 * args.track_buffer)
        self.kalman_filter = self.get_kalmanfilter()
        self.reset_id()

    def update(self, results, filter_cls = None, img=None, fps=None):
        """Updates object tracker with new detections and returns tracked object bounding boxes."""
        self.frame_id += 1
        activated_stracks = []
        refined_stracks = []
        lost_stracks = []
        removed_stracks = []
        filtered_stracks = []

        scores = results.conf
        bboxes = results.xyxy
        # Add index
        bboxes = np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1)
        cls = results.cls

        if filter_cls:
            remain_inds = scores > self.args.track_high_thresh and cls != filter_cls
            inds_low = scores > self.args.track_low_thresh and cls != filter_cls
            inds_high = scores < self.args.track_high_thresh and cls != filter_cls
            filter_index = cls == filter_cls
        
        else:
            remain_inds = scores > self.args.track_high_thresh
            inds_low = scores > self.args.track_low_thresh
            inds_high = scores < self.args.track_high_thresh
            filter_index = scores < 0 # Always False, no detections from Motion_Detector to be filtered

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]
        cls_keep = cls[remain_inds]
        cls_second = cls[inds_second]
        dets_filter = bboxes[filter_index]
        scores_filter = scores[filter_index]
        cls_filter = cls[filter_index]

        detections = self.init_track(dets, scores_keep, cls_keep, img)
        # Add newly detected tracklets to tracked_stracks
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        # Step 2: First association, with high score detection boxes
        strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        self.multi_predict(strack_pool)
        if hasattr(self, 'gmc') and img is not None:
            warp = self.gmc.apply(img, dets)
            STrack.multi_gmc(strack_pool, warp)
            STrack.multi_gmc(unconfirmed, warp)

        dists = self.get_dists(strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refined_stracks.append(track)

        # Step 3: Second association, with low score detection boxes
        # association the untrack to the low score detections
        detections_second = self.init_track(dets_second, scores_second, cls_second, img)
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refined_stracks.append(track)

        # Step 4: Process detections that need to be filtered
        detections_filter = self.init_track(dets_filter, scores_filter, cls_filter, img)
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_filter)
        matches, u_track, u_detection_filter = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_filter[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refined_stracks.append(track)


        for it in u_track:
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # Match the remaining detections that require filtering with the motion_history list.
        # When there at least 5 frames of history, determine speed, direction and initialise track depending on results
        # If less than 5 frame, update motion_history
        # If more than 5 but not object, discard the object
        ftrack_pool = self.filtered_stracks
        self.multi_predict(ftrack_pool)

        if hasattr(self, 'gmc') and img is not None:
            warp = self.gmc.apply(img, dets_filter)
            STrack.multi_gmc(ftrack_pool, warp)

        filter_stracks = [ftrack_pool[i] for i in u_track if ftrack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(filter_stracks, u_detection_filter)
        matches, _, u_detection_filter = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = filter_stracks[itracked]
            det = u_detection_filter[idet]
            if track.tracklet_len <= self.args.frame_filter:
                track.update(det, self.frame_id)
                filtered_stracks.append(track)
            else:
                # TODO: filtering using previous location
                # This will need to store the previous locations, not only the first previous one
                # Given motion, compute Speed, direction -> filter (use area of box to adapt threshold)
                positions = track.positions

                motion_filter = self.compute_motion_metrics(positions, fps)

                if motion_filter['Curvature']  > self.args.curvature_threshold or motion_filter['Smoothness'] < self.args.smoothness_threshold:
                    if track.state == TrackState.Tracked:
                        track.update(det, self.frame_id)
                        activated_stracks.append(track)
                    else:
                        track.re_activate(det, self.frame_id, new_id=False)
                        refined_stracks.append(track)
                    continue

                if fps:
                    if motion_filter['Velocity'] > self.args.velocity_threshold:
                        if track.state == TrackState.Tracked:
                            track.update(det, self.frame_id)
                            activated_stracks.append(track)
                        else:
                            track.re_activate(det, self.frame_id, new_id=False)
                            refined_stracks.append(track)
                        continue
                
                filtered_stracks.append(track)

    

        # Deal with unconfirmed tracks, usually tracks with only one beginning frame
        detections = [detections[i] for i in u_detection]
        dists = self.get_dists(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # Step 5: Init new stracks
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.args.new_track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)    

        detections = [detections[i] for i in u_detection_second]
        for inew in u_detection_second:
            track = detections[inew]
            if track.score < self.args.new_track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)    

        # Step 6: Update state
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refined_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.removed_stracks)
        self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.filtered_stracks = filtered_stracks
        if len(self.removed_stracks) > 1000:
            self.removed_stracks = self.removed_stracks[-999:]  # clip remove stracks to 1000 maximum
        return np.asarray(
            [x.tlbr.tolist() + [x.track_id, x.score, x.cls, x.idx] for x in self.tracked_stracks if x.is_activated],
            dtype=np.float32)

    def get_kalmanfilter(self):
        """Returns a Kalman filter object for tracking bounding boxes."""
        return KalmanFilterXYAH()

    def init_track(self, dets, scores, cls, img=None):
        """Initialize object tracking with detections and scores using STrack algorithm."""
        return [STrack(xyxy, s, c) for (xyxy, s, c) in zip(dets, scores, cls)] if len(dets) else []  # detections

    def get_dists(self, tracks, detections):
        """Calculates the distance between tracks and detections using IOU and fuses scores."""
        dists = matching.iou_distance(tracks, detections)
        # TODO: mot20
        # if not self.args.mot20:
        dists = matching.fuse_score(dists, detections)
        return dists

    def multi_predict(self, tracks):
        """Returns the predicted tracks using the YOLOv8 network."""
        STrack.multi_predict(tracks)

    def reset_id(self):
        """Resets the ID counter of STrack."""
        STrack.reset_id()

    @staticmethod
    def joint_stracks(tlista, tlistb):
        """Combine two lists of stracks into a single one."""
        exists = {}
        res = []
        for t in tlista:
            exists[t.track_id] = 1
            res.append(t)
        for t in tlistb:
            tid = t.track_id
            if not exists.get(tid, 0):
                exists[tid] = 1
                res.append(t)
        return res

    @staticmethod
    def sub_stracks(tlista, tlistb):
        track_ids_b = {t.track_id for t in tlistb}
        return [t for t in tlista if t.track_id not in track_ids_b]

    @staticmethod
    def remove_duplicate_stracks(stracksa, stracksb):
        """Remove duplicate stracks with non-maximum IOU distance."""
        pdist = matching.iou_distance(stracksa, stracksb)
        pairs = np.where(pdist < 0.15)
        dupa, dupb = [], []
        for p, q in zip(*pairs):
            timep = stracksa[p].frame_id - stracksa[p].start_frame
            timeq = stracksb[q].frame_id - stracksb[q].start_frame
            if timep > timeq:
                dupb.append(q)
            else:
                dupa.append(p)
        resa = [t for i, t in enumerate(stracksa) if i not in dupa]
        resb = [t for i, t in enumerate(stracksb) if i not in dupb]
        return resa, resb
    
    @staticmethod
    def compute_motion_metrics(positions, fps=None):
        """
        
        Input: 
            - points - list of (x, y) coordinates representing the trajectory
        
        Output: 
            - Dict[str, float]:
                - Curvature (float): Mean curvature of the trajectory
                - Smoothness (float): Mean smoothness of the trajectory
                - Dir_changes (int): Number of direction changes
                - Velocity (float): Velocity of the tracked object. Only when fps is given
        """

        curvature = []
        smoothness = []
        num_changes = 0
        velocity = []

        # Iterate over the points in the trajectory
        for i in range(1, len(positions) - 1):
            # Compute Curvature
            # Calculate the vectors between the current point and its neighbors
            vector1 = np.array(positions[i][:2]) - np.array(positions[i-1][:2])
            vector2 = np.array(positions[i+1][:2]) - np.array(positions[i][:2])

            # Calculate the cross product between the vectors
            cross_product = np.cross(vector1, vector2)

            # Calculate the magnitude of the cross product divided by the product of the vector magnitudes
            magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
            curvature.append(cross_product / magnitude)

            # Compute Smoothness
            # Calculate the distances between the current point and its neighbors
            dist1 = distance.euclidean(positions[i][:2], positions[i-1][:2])
            dist2 = distance.euclidean(positions[i][:2], positions[i+1][:2])

            # Calculate the ratio of the smaller distance to the larger distance
            ratio = min(dist1, dist2) / max(dist1, dist2)
            smoothness.append(ratio)

            # Calculate the dot product between the vectors
            dot_product = np.dot(vector1, vector2)

            # If the dot product is negative, it indicates a direction change
            if dot_product < 0:
                num_changes += 1

            if fps:
                # Calculate the velocity as displacement divided by time
                velocity.append(dist1 / fps)
        
        if fps:
            velocity.append(distance.euclidean(positions[-1][:2], positions[-2][:2]))

        return {'Curvature': np.mean(curvature), 'Smoothness': np.mean(smoothness), "Dir_changes": num_changes, "Velocity": np.mean(velocity) if fps else None}
