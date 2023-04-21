import os,sys,pdb,torch
f0up_key=sys.argv[1]
input_path=sys.argv[2]
index_path=sys.argv[3]
npy_path=sys.argv[4]
opt_path=sys.argv[5]
model_path=sys.argv[6]
print(sys.argv)
sys.argv=['myinfer.py']
now_dir=os.getcwd()
sys.path.append(now_dir)
from vc_infer_pipeline import VC
from infer_pack.models import SynthesizerTrnMs256NSFsid, SynthesizerTrnMs256NSFsid_nono
from my_utils import load_audio
from fairseq import checkpoint_utils
from scipy.io import wavfile


# f0up_key=0
# input_path=r"E:\codes\py39\RVC-beta\todo-songs\1111.wav"
# index_path=r"E:\codes\py39\logs\mi-test\added_IVF677_Flat_nprobe_7.index"
# npy_path  =r"E:\codes\py39\logs\mi-test\total_fea.npy"
# opt_path  ="test.wav"
# model_path="mi-test.pth"



hubert_model=None
is_half=False
device="cuda"
def load_hubert():
    global hubert_model
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(["hubert_base.pt"],suffix="",)
    hubert_model = models[0]
    hubert_model = hubert_model.to(device)
    if(is_half):hubert_model = hubert_model.half()
    else:hubert_model = hubert_model.float()
    hubert_model.eval()


def vc_single(sid,input_audio,f0_up_key,f0_file,f0_method,file_index,file_big_npy,index_rate):#spk_item, input_audio0, vc_transform0,f0_file,f0method0
    global tgt_sr,net_g,vc,hubert_model
    if input_audio is None:return "You need to upload an audio", None
    f0_up_key = int(f0_up_key)
    audio=load_audio(input_audio,16000)
    times = [0, 0, 0]
    if(hubert_model==None):load_hubert()
    if_f0 = cpt.get("f0", 1)
    audio_opt=vc.pipeline(hubert_model,net_g,sid,audio,times,f0_up_key,f0_method,file_index,file_big_npy,index_rate,if_f0,f0_file=f0_file)
    print(times)
    return audio_opt


def get_vc(sid):
    global n_spk,tgt_sr,net_g,vc,cpt
    person = "weights/%s" % (sid)
    print("loading %s"%person)
    cpt = torch.load(person, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3]=cpt["weight"]["emb_g.weight"].shape[0]#n_spk
    if_f0=cpt.get("f0",1)
    if(if_f0==1):
        net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=is_half)
    else:
        net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    del net_g.enc_q
    print(net_g.load_state_dict(cpt["weight"], strict=False))  # 不加这一行清不干净，真奇葩
    net_g.eval().to(device)
    if (is_half):net_g = net_g.half()
    else:net_g = net_g.float()
    vc = VC(tgt_sr, device, is_half)
    n_spk=cpt["config"][-3]
    # return {"visible": True,"maximum": n_spk, "__type__": "update"}


get_vc(model_path)
wav_opt=vc_single(0,input_path,f0up_key,None,"harvest",index_path,npy_path,0.6)
wavfile.write(opt_path, tgt_sr, wav_opt)

