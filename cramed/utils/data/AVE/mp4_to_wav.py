import os
import moviepy.editor as mp

def extract_audio(videos_file_path):
    my_clip = mp.VideoFileClip(videos_file_path)
    return my_clip


# train_videos = r'D:\yunfeng\data\AVE_Dataset\trainSet.txt'
# test_videos = r'D:\yunfeng\data\AVE_Dataset\testSet.txt'
#
# train_audio_dir = r'D:\yunfeng\data\AVE_Dataset\Audios\train_set'
# test_audio_dir = r'D:\yunfeng\data\AVE_Dataset\Audios\test_set'
# if not os.path.exists(train_audio_dir):
#     os.makedirs(train_audio_dir)
# if not os.path.exists(test_audio_dir):
#     os.makedirs(test_audio_dir)
#
#
# # test set processing
# with open(test_videos, 'r') as f:
#     files = f.readlines()
#
# for i, item in enumerate(files):
#     if i % 500 == 0:
#         print('*******************************************')
#         print('{}/{}'.format(i, len(files)))
#         print('*******************************************')
#     item = item.split('&')
#     mp4_filename = os.path.join(r'D:\yunfeng\data\AVE_Dataset\AVE', item[1]+'.mp4')
#     wav_filename = os.path.join(test_audio_dir, item[1]+'&'+item[0]+'.wav')
#     if os.path.exists(wav_filename):
#         pass
#     else:
#         my_clip = extract_audio(mp4_filename)
#         my_clip.audio.write_audiofile(wav_filename)
#
#         #os.system('ffmpeg -i {} -acodec pcm_s16le -ar 16000 {}'.format(mp4_filename, wav_filename))
#
# # train set processing
# with open(train_videos, 'r') as f:
#     files = f.readlines()
#
# for i, item in enumerate(files):
#     if i % 500 == 0:
#         print('*******************************************')
#         print('{}/{}'.format(i, len(files)))
#         print('*******************************************')
#     item = item.split('&')
#     mp4_filename = os.path.join(r'D:\yunfeng\data\AVE_Dataset\AVE', item[1] + '.mp4')
#     wav_filename = os.path.join(train_audio_dir, item[1]+'&'+item[0]+'.wav')
#     if os.path.exists(wav_filename):
#         pass
#     else:
#         my_clip = extract_audio(mp4_filename)
#         my_clip.audio.write_audiofile(wav_filename)
#
#         #os.system('ffmpeg -i {} -acodec pcm_s16le -ar 16000 {}'.format(mp4_filename, wav_filename))



all_videos = r'D:\yunfeng\data\AVE_Dataset\Annotations.txt'
all_audio_dir = r'D:\yunfeng\data\AVE_Dataset\Audios'
if not os.path.exists(all_audio_dir):
    os.makedirs(all_audio_dir)

# train set processing
with open(all_videos, 'r') as f:
    files = f.readlines()

for i, item in enumerate(files[1:]):
    if i % 500 == 0:
        print('*******************************************')
        print('{}/{}'.format(i, len(files)))
        print('*******************************************')
    item = item.split('&')
    mp4_filename = os.path.join(r'D:\yunfeng\data\AVE_Dataset\AVE', item[1] + '.mp4')
    wav_filename = os.path.join(all_audio_dir, item[1]+'.wav')
    if os.path.exists(wav_filename):
        pass
    else:
        my_clip = extract_audio(mp4_filename)
        my_clip.audio.write_audiofile(wav_filename)

        #os.system('ffmpeg -i {} -acodec pcm_s16le -ar 16000 {}'.format(mp4_filename, wav_filename))

