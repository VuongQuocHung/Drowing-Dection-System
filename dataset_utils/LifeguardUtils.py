import csv
import glob
import os
import shutil
import subprocess
import zipfile
from PIL import Image

import pandas as pd
import cv2
import xml.etree.ElementTree as ET
from yt_dlp import YoutubeDL, DownloadError


class LifeguardUtils:
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    #lifeguard_folder = os.path.join(current_file_directory, '..', 'dataset_lifeguard')
    lifeguard_folder = os.path.join('./dataset_lifeguard')
    utils_folder = os.path.join(current_file_directory)

    frames_folder = os.path.join(lifeguard_folder, 'frames')
    labels_folder = os.path.join(lifeguard_folder, 'labels')
    videos_folder = os.path.join(lifeguard_folder, 'videos')
    videos_source_folder = os.path.join(lifeguard_folder, 'videos_source')
    datalist_file = os.path.join(lifeguard_folder, 'datalist.csv')

    splitlists_folder = os.path.join(utils_folder, 'splitlists')
    # defaults for writing to
    testlist_file = os.path.join(splitlists_folder, 'test.csv')
    trainlist_file = os.path.join(splitlists_folder, 'train.csv')

    @staticmethod
    def snippet_name(video_id, start, end, x, y, width):
        start = str(start)[:-3].replace(':', '-')
        end = str(end)[:-3].replace(':', '-')
        x = str(x)[:-2].replace('.', '-')
        y = str(y)[:-2].replace('.', '-')
        width = str(width)[:-2].replace('.', '-')
        return video_id + '_' + start + '--' + end + '_' + x + '--' + y + '--' + width


class DatalistToFiles(LifeguardUtils):

    @classmethod
    def download_lifeguard_rescue_yt_channel(cls):
        os.makedirs(cls.videos_source_folder, exist_ok=True)
        options = {
            'format': 'bv',
            'paths': {'home': cls.videos_source_folder},
        }
        ydl = YoutubeDL(options)
        try:
            ydl.download(['https://www.youtube.com/@LifeguardRescue/videos'])
        except DownloadError:
            print("An error occurred while downloading the videos.")

    @classmethod
    def create_videos_and_frames(cls):
        """no need to run download_lifeguard_rescue_yt_channel before this function,
        because this function will download any video to videos_source, which is not yet downloaded."""

        with open(cls.datalist_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                [video_id, start, end, x, y, width, _] = row

                # if this function is initially used to create the complete dataset, we first need to download the videos_source
                videos_source_files = glob.glob(os.path.join(cls.videos_source_folder, f"*{video_id}*"))
                if videos_source_files:
                    videos_source_file = videos_source_files[0]
                else:
                    os.makedirs(cls.videos_source_folder, exist_ok=True)
                    options = {
                        'format': 'bv',
                        'paths': {'home': cls.videos_source_folder},
                    }
                    ydl = YoutubeDL(options)
                    try:
                        ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
                        videos_source_file = glob.glob(os.path.join(cls.videos_source_folder, f"*{video_id}*"))[0]
                    except DownloadError:
                        print("An error occurred while downloading the video.")
                        continue

                _, file_extension = os.path.splitext(videos_source_file)
                snippet = cls.snippet_name(video_id, start, end, x, y, width)

                cap = cv2.VideoCapture(videos_source_file)
                video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                x_px = int(float(x) * video_width)
                y_px = int(float(y) * video_height)
                # if, because of rounding, region for snipping is outside of video, ffmpeg automatically moves the area
                # for snipping back inside the video. but if snipping height or width is bigger than video,
                # there will be an error. that is why we do the min here.
                width_px = min(int(float(width) * video_height), video_width, video_height)
                cap.release()

                # update videos
                target = os.path.join(cls.videos_folder, f"{snippet}{file_extension}")
                if not os.path.exists(target):
                    os.makedirs(cls.videos_folder, exist_ok=True)
                    ffmpeg_videos = [
                        'ffmpeg',
                        '-i', videos_source_file,
                        '-ss', start,
                        '-to', end,
                        '-vf', f"crop={width_px}:{width_px}:{x_px}:{y_px},scale=224:224,fps=30",
                        '-preset', 'veryslow',
                        '-crf', '17',
                        '-b:v', '0',
                        target
                    ]
                    subprocess.run(ffmpeg_videos)

                # update frames
                frames_snippet_folder = os.path.join(cls.frames_folder, snippet)
                target = os.path.join(frames_snippet_folder, "%06d.jpg")
                if not os.path.exists(frames_snippet_folder):
                    os.makedirs(frames_snippet_folder, exist_ok=True)
                    ffmpeg_frames = [
                        'ffmpeg',
                        '-i', videos_source_file,
                        '-ss', start,
                        '-to', end,
                        '-vf', f"crop={width_px}:{width_px}:{x_px}:{y_px},scale=224:224,fps=30",
                        '-q:v', '1',
                        target
                    ]
                    subprocess.run(ffmpeg_frames)

    @classmethod
    def remove_entry(cls, snippet):
        # remove entry from datalist_file
        with open(cls.datalist_file, 'r') as f:
            data = list(csv.reader(f))
        data = [row for row in data if cls.snippet_name(*row[:6]) != snippet]
        with open(cls.datalist_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(data)

        # remove files from frames
        frames_snippet_folder = os.path.join(cls.frames_folder, snippet)
        if os.path.exists(frames_snippet_folder):
            shutil.rmtree(frames_snippet_folder)

        # remove file from labels
        label_file = os.path.join(cls.labels_folder, f'{snippet}.xml')
        if os.path.exists(label_file):
            os.remove(label_file)

        # remove file from videos
        for file in glob.glob(os.path.join(cls.videos_folder, f'{snippet}*')):
            if os.path.exists(file):
                os.remove(file)


class DataListToSplitlists:
    
    @classmethod
    def split_data(cls, ratio=0.90, train_only_snippets=[]):
        # load the datalist.csv
        df = pd.read_csv(cls.datalist_file, header=None, dtype=str)
        # filter only rows with 'y' and existing label file
        df = df[df.iloc[:, -1] == 'y']
        # apply the snippet_name function to each row
        df["snippet"] = df.apply(lambda row: cls.snippet_name(*row[:6]), axis=1)
        # check if the label file exists in cls.labels_folder for each row
        df = df[df["snippet"].apply(lambda x: os.path.isfile(os.path.join(cls.labels_folder, f"{x}.xml")))]

        # add a new column indicating whether each snippet is in train_only_snippets
        df["train_only"] = df["snippet"].apply(lambda x: x in train_only_snippets)

        # shuffle the dataframe for randomness
        df = df.sample(frac=1).reset_index(drop=True)
        # separate out the train_only rows
        df_train_only = df[df["train_only"] == True]
        # remove the train_only rows from df
        df = df[df["train_only"] == False]
        # calculate the number of remaining training data points
        num_train = int(len(df) * ratio)
        # ensure at least one sample for test dataset
        num_train = min(num_train, len(df) - 1)
        # split the remaining dataframe into training and testing sets
        df_train = pd.concat([df.iloc[:num_train], df_train_only])
        df_test = df.iloc[num_train:].copy()

        # create a new column for df_test to indicate if drowning occurs
        df_test.loc[:, "label"] = df_test["snippet"].apply(
            lambda x: 'd' if cls.snippet_contains_drown(x) else 's')

        # write the training and testing data to CSV files
        df_train['snippet'].to_csv(cls.trainlist_file, index=False, header=False)
        df_test[['snippet', 'label']].to_csv(cls.testlist_file, index=False, header=False)


    @classmethod
    def snippet_contains_drown(cls, snippet):
        label_file = os.path.join(cls.labels_folder, f'{snippet}.xml')

        tree = ET.parse(label_file)
        root = tree.getroot()

        for image in root.iter('image'):
            for box in image.iter('box'):
                if box.get('label') == 'drown':
                    return True
        return False

    @classmethod
    def sort(cls):
        with open(cls.datalist_file, 'r', newline='') as f:
            reader = csv.reader(f)
            data = list(reader)

        data.sort(key=lambda row: tuple(cell.lower() if isinstance(cell, str) else cell for cell in row))

        with open(cls.datalist_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(data)

    @classmethod
    def extract_xml_from_zip(cls):
        # traverse through directory
        for filename in os.listdir(cls.labels_folder):
            # check if file is a zip file
            if filename.endswith('.zip'):
                zip_file_path = os.path.join(cls.labels_folder, filename)
                # open the zip file
                with zipfile.ZipFile(os.path.join(cls.labels_folder, filename), 'r') as zip_ref:
                    # get list of all files names in zip
                    zip_files = zip_ref.namelist()

                    # check that there's exactly one file in the zip
                    if len(zip_files) != 1:
                        print(f"Skipping {filename} because it contains more than one file")
                        continue

                    # extract the file
                    xml_data = zip_ref.read(zip_files[0])

                    # write the file to a new XML file with the same name as the zip file
                    with open(os.path.join(cls.labels_folder, filename.replace('.zip', '.xml')), 'wb') as f:
                        f.write(xml_data)

                os.remove(zip_file_path)

    @classmethod
    def append_drown_column(cls, testlist_file):
        # load the testlist_file
        df = pd.read_csv(testlist_file, header=None, dtype=str, names=["snippet"])

        # create a new column to indicate if drowning occurs
        df.loc[:, "label"] = df["snippet"].apply(
            lambda x: 'd' if cls.snippet_contains_drown(x) else 's')

        # write the labeled data back to the same testlist_file
        df.to_csv(testlist_file, index=False, header=False)

    @classmethod
    def datalist_to_testlist(cls, datalist):
         # load the datalist.csv
         df = pd.read_csv(datalist, header=None, dtype=str)
         # apply the snippet_name function to each row
         df["snippet"] = df.apply(lambda row: cls.snippet_name(*row[:6]), axis=1)

         df['snippet'].to_csv(datalist, index=False, header=False)


class FileLoader(LifeguardUtils):

    @classmethod
    def get_num_frames(cls, snippet):
        frames_snippet_folder = os.path.join(cls.frames_folder, snippet)
        return len(glob.glob(os.path.join(frames_snippet_folder, '*')))

    @classmethod
    def get_video_clip(cls, snippet, seq):
        video_clip = []
        for i in seq:
            img_path = os.path.join(cls.frames_folder, snippet, '{:06}.jpg'.format(i))
            frame = Image.open(img_path).convert('RGB')
            video_clip.append(frame)
        return video_clip

    @staticmethod
    def get_box_and_label(box):
        xtl = float(box.get('xtl'))
        ytl = float(box.get('ytl'))
        xbr = float(box.get('xbr'))
        ybr = float(box.get('ybr'))
        label_mapping = {'swim': 0, 'drown': 1}
        label = label_mapping[box.get('label')]
        return [xtl, ytl, xbr, ybr, label]

    @classmethod
    def get_box_and_label_list(cls, snippet, frame):
        label_file = os.path.join(cls.labels_folder, f'{snippet}.xml')
        box_and_label_list = []
        if not os.path.exists(label_file):
            return box_and_label_list
        tree = ET.parse(label_file)
        root = tree.getroot()

        for image in root.iter('image'):
            if int(image.get('id')) == frame - 1:  # adjusting frame numbers as they start from 0 in annotation file
                for box in image.iter('box'):
                    box_and_label_list.append(cls.get_box_and_label(box))

        return box_and_label_list


# DatalistToFiles.download_lifeguard_rescue_yt_channel()
DatalistToFiles.create_videos_and_frames()
# DatalistToFiles.remove_entry('zufy4aBEY_00-00-08--00-01-11_0-27--0-34--0-65')

# DataListToSplitlists.split_data(0, ['zuZIfy4aBEY_00-00-08--00-01-11_0-27--0-34--0-65',
#                                       'vm1SOtbe-JE_00-00-14--00-01-10_0-06--0-36--0-28',
#                                       'usWABZrI75M_00-00-00--00-00-56_0-41--0-19--0-24',
#                                       'lvS0S6X3_xc_00-00-00--00-01-15_0-00--0-40--0-59'])
# DataListToSplitlists.sort()
# DataListToSplitlists.extract_xml_from_zip()
# DataListToSplitlists.append_drown_column(os.path.join(LifeguardUtils.splitlists_folder, '9_test.csv'))
# DataListToSplitlists.datalist_to_testlist(os.path.join(LifeguardUtils.splitlists_folder, 'difficult_test.csv'))
