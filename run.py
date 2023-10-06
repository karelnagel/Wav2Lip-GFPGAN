import os
import cv2
from tqdm import tqdm
from subprocess import call
import sys

if (len(sys.argv) < 4):
  print("Usage: python run.py <inputAudioPath> <inputVideoPath> <outputPath>")
  exit()

inputAudioPath = sys.argv[1]
inputVideoPath = sys.argv[2]
outputPath = sys.argv[3]

wav2lipFolderName = 'Wav2Lip-master'
gfpganFolderName = 'GFPGAN-master'
lipSyncedOutputPath = outputPath+'/result.mp4'

if not os.path.exists(outputPath):
  os.makedirs(outputPath)

call(['python', f'{wav2lipFolderName}/inference.py', '--checkpoint_path', f'{wav2lipFolderName}/checkpoints/wav2lip.pth', '--face', inputVideoPath, '--audio', inputAudioPath, '--outfile', lipSyncedOutputPath])

inputVideoPath = outputPath+'/result.mp4'
unProcessedFramesFolderPath = outputPath+'/frames'

if not os.path.exists(unProcessedFramesFolderPath):
  os.makedirs(unProcessedFramesFolderPath)

vidcap = cv2.VideoCapture(inputVideoPath)
numberOfFrames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = vidcap.get(cv2.CAP_PROP_FPS)
print("FPS: ", fps, "Frames: ", numberOfFrames)

for frameNumber in tqdm(range(numberOfFrames)):
    _,image = vidcap.read()
    cv2.imwrite(os.path.join(unProcessedFramesFolderPath, str(frameNumber).zfill(4)+'.jpg'), image)

restoredFramesPath = outputPath + '/restored_imgs/'
os.chdir(gfpganFolderName)
call(['python', 'inference_gfpgan.py', '-i', f"../{unProcessedFramesFolderPath}", '-o', f"../{outputPath}", '-v', '1.3', '-s', '2', '--only_center_face', '--bg_upsampler', 'None'])
os.chdir('..')

# putting the frames and audio together into mp4
finalProcessedOuputVideo = outputPath+'/final_with_audio.mp4'
call(['ffmpeg', '-y', '-r', '30', '-i', restoredFramesPath + '/%04d.jpg', '-i', inputAudioPath, '-map', '0:v', '-map', '1:a', '-c:v', 'h264_nvenc', '-vf', 'fps=30', '-pix_fmt', 'yuv420p', '-shortest', finalProcessedOuputVideo])
