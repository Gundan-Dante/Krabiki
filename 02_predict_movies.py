#!/usr/bin/env python
# coding: utf-8

# In[1]:


import net2brain
import cv2
import os, glob, json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import decimate


# In[2]:


SUB = 1


# First we load the model trained in the `01_preprocess_and_train_model.ipynb` notebook

# In[3]:


import joblib
B = joblib.load(f'encoder_sub-{SUB:02d}.joblib')
sc, pca, sc2, model = B['sc'], B['pca'], B['sc2'], B['model']
LAYER, n_chan, n_time = B['LAYER'], B['n_chan'], B['n_time']
DEC, SFREQ, N_FRAMES = B['DEC'], B['SFREQ'], B['N_FRAMES']


# In[4]:


# load test data
TEST_DIR  = './tst'
STIM_TEST_DIR   = './stimuli_test'
FRAME_TEST_DIR = './frames_test'
FEAT_TEST_DIR = './features_test'


# In[5]:


tst_eeg = np.load(f'{TEST_DIR}/sub-{SUB:02d}_tst.npy')


# In[6]:


tst_eeg.shape


# In[7]:


tst_eeg = decimate(tst_eeg, DEC, axis=-1, ftype='fir', zero_phase=True)

times = np.arange(tst_eeg.shape[-1]) / (SFREQ / DEC)      # 0 .. 3 s
n_stim, n_chan, n_time = tst_eeg.shape
print(tst_eeg.shape)


# In[ ]:


from net2brain.feature_extraction import FeatureExtractor

fx = FeatureExtractor(model='AlexNet', netset='Standard', device='cpu')   # 'cuda' if available


# In[11]:


vids = sorted(glob.glob(os.path.join(STIM_TEST_DIR, '*.mp4')))
print(len(vids), 'videos found')
print(vids[:3], '...', vids[-2:])


# In[12]:


def video_for(stim_id):
    positions = [vid for vid in vids if f'/{stim_id}.mp4' in vid]
    return positions[0]

def dump_frames(stim_ids, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for sid in stim_ids:
        cap = cv2.VideoCapture(video_for(sid))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        picks = np.linspace(0, max(total - 1, 0), N_FRAMES).astype(int)
        for k, fno in enumerate(picks):
            path = os.path.join(out_dir, f'{int(sid):05d}_f{k:02d}.jpg')
            if os.path.exists(path):
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fno))
            ok, frame = cap.read()
            if ok:
                cv2.imwrite(path, frame)
        cap.release()


# In[11]:


tst_ids = [ii for ii in range(1001,1103)]


# In[ ]:


dump_frames(np.array(tst_ids), FRAME_TEST_DIR)
print(len(glob.glob(f'{FRAME_TEST_DIR}/*.jpg')), 'frames written')


# In[ ]:


fx.extract(data_path=FRAME_TEST_DIR, layers_to_extract=[LAYER], save_path=FEAT_TEST_DIR)


# In[13]:


def features_for(ids, layer_path):
    z = np.load(+FEAT_TEST_DIR+'/consolidated_'+layer_path+'.npz')
    rows = []
    for sid in ids:
        keys = [k for k in z.files if k.startswith(f'{int(sid):05d}_')]
        assert keys, f'no features for stimulus {sid}'
        rows.append(np.mean([z[k].flatten() for k in keys], axis=0))
    return np.stack(rows)

X_tst = features_for(tst_ids, LAYER)
print(X_tst.shape)


# Map the test data into the same feature space the model learned

# In[14]:


X_tst_z = sc.transform(X_tst)
X_tst_p = pca.transform(X_tst_z)
X_tst_p.shape


# In[15]:


X_tst_s = sc2.transform(X_tst_p)
pred_tst_eeg = model.predict(X_tst_s)


# In[16]:


rehsaped_pred_tst_eeg = pred_tst_eeg.reshape(len(tst_ids), n_chan, n_time)


# In[17]:


rehsaped_pred_tst_eeg.shape


# In[18]:


gt = np.load(f'./tst_stim/sub-{SUB:02d}_tst_stim.npy')
order = np.argsort(gt)                 # positions that sort gt ascending


# ### Predict top 1 and top 5

# In[19]:


M = tst_eeg[order][:, :, 5:60].reshape(102, -1)
P = rehsaped_pred_tst_eeg[:, :, 5:60].reshape(102, -1)
Mz = (M - M.mean(1, keepdims=True)) / M.std(1, keepdims=True)
Pz = (P - P.mean(1, keepdims=True)) / P.std(1, keepdims=True)

S = Mz @ Pz.T / M.shape[1]                          # (102, 102) similarity
match = S.argmax(1)                                 # best predicted row per measured row


# In[20]:


t = np.arange(102)
ranks = (-S).argsort(1).argsort(1)[t, t] + 1
print('top-1', (ranks == 1).mean(), 'top-5', (ranks <= 5).mean(), 'median', np.median(ranks))


# Seems low, but consider that chance classifier gets:
# top-1: 1/102 = 0.0098 (0.98%)
# top-5: 5/102 = 0.049 (4.9%)
# mean rank = (102+1)/2 = 51.5
# 
# So the results of this simple baseline are already 4x better than chance. Using better models, more s

# In[36]:


video_ids = np.array(tst_ids)              # column j -> video 1001+j
order_idx = np.argsort(-S, axis=1)          # (102, 102) column indices, best first
ranked = video_ids[order_idx]               # (102, 102) video IDs, best first


# Here `ranked` is the ranked candidate list, which for each eeg sample (row) ranks the videos. Please send these ranked candidate lists for subjects 5 and 6

# In[41]:


ranked


# In[ ]:


np.save(f'sub-0{subject_num}_ranked_candidate_list.npy', eeg_arr)


# In[42]:


truth_ids = gt[order]                       # row i's actual video

top1 = (ranked[:, 0] == truth_ids).mean()
top5 = (ranked[:, :5] == truth_ids[:, None]).any(axis=1).mean()

print(f'top-1 {top1:.4f}  (chance {1/102:.4f})')
print(f'top-5 {top5:.4f}  (chance {5/102:.4f})')


# In[ ]:




