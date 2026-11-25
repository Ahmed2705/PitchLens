import unittest
import numpy as np
from pitchlens.team_classification import TeamClassifier, TeamConfig, jersey_colour
from pitchlens.tracking import FrameTracks, Track

class TeamTests(unittest.TestCase):
    def test_grass_and_small_crop_unknown(self):
        self.assertIsNone(jersey_colour(np.full((60,60,3), (0,180,0), np.uint8), (15,15,45,45)))
        self.assertIsNone(jersey_colour(np.zeros((2,2,3), np.uint8), (0,0,2,2)))
        frame = np.full((100,100,3), 255, np.uint8)
        frame[20:80,20:80] = (0,180,0)
        self.assertIsNotNone(jersey_colour(frame, (20,20,80,80)))

    def test_clustering_exclusion_and_reset(self):
        frame = np.zeros((100, 300, 3), np.uint8)
        tracks = []
        for i in range(6):
            x = i*45
            frame[10:90,x:x+35] = (255,0,0) if i < 3 else (255,255,255)
            tracks.append(Track((x,10,x+35,90),0,i,.9,True))
        tracks.extend([Track((0,10,35,90),2,20,.9,True), Track((0,10,35,90),1,21,.9,True)])
        classifier = TeamClassifier()
        result = classifier.update(frame, FrameTracks(0,0,False,tuple(tracks),()))
        self.assertEqual(len(set(result[:3])), 1)
        self.assertEqual(len(set(result[3:6])), 1)
        self.assertNotEqual(result[0], result[3])
        self.assertIsNone(result[6])
        self.assertIsNone(result[7])
        self.assertEqual(classifier.update(frame, FrameTracks(1,1,True,(tracks[0],),())), [None])
        classifier.reset()
        self.assertEqual(classifier.colours, {})

    def test_later_opponents_correct_early_split(self):
        frame = np.zeros((100, 600, 3), np.uint8)
        tracks = []
        for i in range(12):
            x = i*45
            colour = (255,0,0) if i < 3 else ((140,0,0) if i < 6 else (255,255,255))
            frame[10:90,x:x+35] = colour
            tracks.append(Track((x,10,x+35,90),0,i,.9,True))
        classifier = TeamClassifier()
        self.assertEqual(classifier.update(frame, FrameTracks(0,0,False,tuple(tracks[:6]),())), [None]*6)
        result = classifier.update(frame, FrameTracks(1,0,False,tuple(tracks),()))
        self.assertEqual(len(set(result[:6])), 1)
        self.assertNotEqual(result[0], result[6])

    def test_extreme_shadow_is_unknown_for_new_tracks(self):
        frame = np.zeros((100,300,3), np.uint8)
        tracks = []
        for i in range(6):
            x = i*45
            frame[10:90,x:x+35] = (255,0,0) if i < 3 else (20,0,0)
            tracks.append(Track((x,10,x+35,90),0,i,.9,True))
        self.assertEqual(TeamClassifier().update(frame, FrameTracks(0,0,False,tuple(tracks),())), [None]*6)

    def test_existing_tracks_keep_assignments_in_shadow(self):
        frame = np.zeros((100,300,3), np.uint8)
        tracks = []
        for i in range(6):
            x = i*45
            frame[10:90,x:x+35] = (255,0,0) if i < 3 else (255,255,255)
            tracks.append(Track((x,10,x+35,90),0,i,.9,True))
        classifier = TeamClassifier()
        first = classifier.update(frame, FrameTracks(0,0,False,tuple(tracks),()))
        frame[10:90,:125] = (20,0,0)
        shadow = classifier.update(frame, FrameTracks(1,0,False,tuple(tracks),()))
        self.assertNotIn(None, first)
        self.assertEqual(first, shadow)

    def test_goalkeeper_remains_unknown(self):
        frame = np.full((100, 100, 3), 255, np.uint8)
        goalkeeper = Track((10,10,40,90),1,7,.9,True)
        classifier = TeamClassifier()
        for i in range(35):
            self.assertEqual(classifier.update(frame, FrameTracks(i,0,False,(goalkeeper,),())), [None])

    def test_expiry_and_green_jersey(self):
        frame = np.full((100,100,3), (109,180,109), np.uint8)
        frame[20:80,20:80] = (0,180,0)
        self.assertIsNotNone(jersey_colour(frame, (20,20,80,80)))
        classifier = TeamClassifier(TeamConfig(max_inactive_frames=2))
        t = Track((20,20,80,80),0,4,.9,True)
        classifier.update(frame, FrameTracks(0,0,False,(t,),()))
        classifier.update(frame, FrameTracks(3,0,False,(),()))
        self.assertNotIn(4, classifier.colours)

    def test_history_bounded_and_rejected_fit_unknown(self):
        frame = np.full((80, 50, 3), 255, np.uint8)
        classifier = TeamClassifier()
        for i in range(110):
            t = Track((0,0,40,80),0,i,.9,True)
            self.assertEqual(classifier.update(frame, FrameTracks(i,0,False,(t,),())), [None])
        self.assertLessEqual(len(classifier.colours), 100)
        classifier.centres = np.array([[0,0,0],[255,255,255]], dtype=np.float32)
        self.assertEqual(classifier.update(frame, FrameTracks(111,0,False,(t,),())), [None])

    def test_outlier_cluster_is_not_frozen(self):
        frame = np.zeros((100,480,3),np.uint8)
        tracks=[]
        for i, colour in enumerate([(255,0,0)]*6+[(0,0,255)]*2+[(255,255,255)]*2):
            x=i*45;frame[10:90,x:x+35]=colour
            tracks.append(Track((x,10,x+35,90),0,i,.9,True))
        classifier=TeamClassifier()
        classifier.update(frame,FrameTracks(0,0,False,tuple(tracks),()))
        self.assertIsNone(classifier.centres)
        for i in range(1,33):
            result=classifier.update(frame,FrameTracks(i,0,False,tuple(tracks[:8]),()))
        self.assertNotEqual(result[0],result[6])
        self.assertIsNotNone(result[6])

    def test_neutral_split_is_opt_in(self):
        frame=np.zeros((100,300,3),np.uint8);tracks=[]
        for i in range(6):
            x=i*45;frame[10:90,x:x+35]=(70,70,70) if i<3 else (255,255,255)
            tracks.append(Track((x,10,x+35,90),0,i,.9,True))
        rec=FrameTracks(0,0,False,tuple(tracks),())
        self.assertEqual(TeamClassifier().update(frame,rec),[None]*6)
        out=TeamClassifier(TeamConfig(allow_neutral_split=True)).update(frame,rec)
        self.assertNotEqual(out[0],out[3])
        self.assertNotIn(None,out)

    def test_refit_after_outlier_tracks_expire(self):
        frame=np.zeros((100,600,3),np.uint8);tracks=[]
        for i, colour in enumerate([(255,0,0)]*4+[(0,0,255)]*2+[(255,255,255)]*4):
            x=i*45;frame[10:90,x:x+35]=colour
            tracks.append(Track((x,10,x+35,90),0,i,.9,True))
        clf=TeamClassifier()
        first=clf.update(frame,FrameTracks(0,0,False,tuple(tracks[:6]),()))
        for i in range(1,36):
            out=clf.update(frame,FrameTracks(i,0,False,tuple(tracks[:4]+tracks[6:]),()))
        self.assertEqual(out[0],first[0])
        self.assertNotIn(None,out)
        self.assertNotEqual(out[0],out[4])

    def test_refit_preserves_supported_team_label(self):
        frame=np.zeros((100,600,3),np.uint8); tracks=[]
        for i,colour in enumerate([(255,255,255)]*4+[(255,0,0)]*2+[(0,255,255)]*4):
            x=i*45;frame[10:90,x:x+35]=colour
            tracks.append(Track((x,10,x+35,90),0,i,.9,True))
        clf=TeamClassifier();first=clf.update(frame,FrameTracks(0,0,False,tuple(tracks[:6]),()))
        for i in range(1,33):
            out=clf.update(frame,FrameTracks(i,0,False,tuple(tracks[:4]+tracks[6:]),()))
        self.assertEqual(first[0],out[0])
        self.assertNotIn(None,out)
        self.assertNotEqual(out[0],out[4])

    def test_persistent_unmatched_players_trigger_refit(self):
        frame=np.zeros((100,800,3),np.uint8);tracks=[]
        for i,colour in enumerate([(255,0,0)]*6+[(0,0,255)]*2+[(255,255,255)]*8):
            x=i*45;frame[10:90,x:x+35]=colour
            tracks.append(Track((x,10,x+35,90),0,i,.9,True))
        clf=TeamClassifier();first=clf.update(frame,FrameTracks(0,0,False,tuple(tracks[:8]),()))
        for i in range(1,33):out=clf.update(frame,FrameTracks(i,0,False,tuple(tracks),()))
        self.assertEqual(first[0],out[0])
        self.assertTrue(all(value is not None for value in out[8:]))
        self.assertNotEqual(out[0],out[8])

    def test_rejected_refit_keeps_supported_team(self):
        frame=np.zeros((100,400,3),np.uint8); tracks=[]
        for i,colour in enumerate([(255,0,0)]*4+[(255,255,255)]*2+[(0,0,255),(0,255,255)]):
            x=i*45;frame[10:90,x:x+35]=colour
            tracks.append(Track((x,10,x+35,90),0,i,.9,True))
        clf=TeamClassifier();first=clf.update(frame,FrameTracks(0,0,False,tuple(tracks[:6]),()))
        for i in range(1,33):out=clf.update(frame,FrameTracks(i,0,False,tuple(tracks[:4]+tracks[6:]),()))
        self.assertEqual(out[:4],first[:4])

    def test_lighting_change_recovers_same_team_ids(self):
        tracks=tuple(Track((i*45,10,i*45+35,90),0,i,.9,True) for i in range(6))
        def image(blue,white):
            frame=np.zeros((100,300,3),np.uint8)
            for i in range(6):
                x=i*45;frame[10:90,x:x+35]=(blue,0,0) if i<3 else (white,white,white)
            return frame
        classifier=TeamClassifier()
        initial=classifier.update(image(255,255),FrameTracks(0,0,False,tracks,()))
        for i in range(1,100):
            result=classifier.update(image(125,175),FrameTracks(i,0,False,tracks,()))
        self.assertNotIn(None,initial)
        self.assertEqual(initial,result)

    def test_bad_config(self):
        with self.assertRaises(ValueError): TeamConfig(min_tracks=1)
        with self.assertRaises(ValueError): TeamConfig(min_tracks=101)
        with self.assertRaises(ValueError): TeamConfig(min_separation=float("nan"))

if __name__ == '__main__': unittest.main()
