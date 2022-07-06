import os
import sys
import logging
import shutil
from pathlib import Path
import synapseclient
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils.timer import start_timer, end_timer_and_print, log  # noqa


def convert_m4a_to_wav(m4a_file_path, delete_after_conversion=True):
    import os
    import glob
    from pydub import AudioSegment

    wav_filename = m4a_file_path.replace('.m4a', '.wav')

    try:
        track = AudioSegment.from_file(m4a_file_path,  format='m4a')
        try:
            file_handle = track.export(wav_filename, format='wav')
        except Exception as e:
            logging.error(f"Failed to export {wav_filename}\n")
    except Exception as e:
        logging.error(f"Conversion Failed: could not load {m4a_file_path}\n")


def download_mPower_dataset():

    if(Path("data/dataset/mPower").exists()):
        log("mPower folder found - skipping redownload")
        return

    EMAIL = os.getenv('SYNAPSE_EMAIL')
    API_KEY = os.getenv('SYNAPSE_API_KEY')
    if(not API_KEY or not EMAIL):
        logging.error('EMAIL/API Key not found')
        sys.exit(1)

    syn = synapseclient.Synapse()
    syn.login(email=EMAIL, apiKey=API_KEY, rememberMe=True)

    event_name = "Download mPower Dataset"
    log(event_name)
    start_timer(event_name)

    voice_query = f"select * from syn5511444"
    results = syn.tableQuery(voice_query)

    PD_metadata_query = f'SELECT * FROM syn5511429 where "professional-diagnosis" = 1'
    HC_metadata_query = f'SELECT * FROM syn5511429 where "professional-diagnosis" = 0'
    PD_metadata = syn.tableQuery(PD_metadata_query)
    HC_metadata = syn.tableQuery(HC_metadata_query)

    PD_healthcodes, HC_healthcodes = set(), set()

    for r in PD_metadata:
        PD_healthcodes.add(r[3])
    for r in HC_metadata:
        HC_healthcodes.add(r[3])

    file_handle_to_healthcode_map = {}
    for r in results:
        file_handle_to_healthcode_map[r[-3]] = r[-7]

    # FOR TABLES WITH COLUMNS THAT CONTAIN FILES, WE CAN BULK DOWNLOAD THE FILES AND STORE A MAPPING
    # THE VALUE IN THE TABLE ABOVE IS CALLED A fileHandleId WHICH REFERENCES A FILE THAT CAN BE
    # ACCESSED PROGRAMMATICALLY GET THE FILES THAT CONTAIN VOICE SAMPLES FROM THE AAAAAAh VOICE RECORDING EXERCISE

    file_map = syn.downloadTableColumns(
        results, ['audio_audio.m4a'], downloadLocation="./")
    for file_handle_id, path in file_map.items():
        healthcode = file_handle_to_healthcode_map.get(file_handle_id)

        output_file_path = ["data/dataset/mPower/"]

        if(healthcode in PD_healthcodes):
            output_file_path.append("PD/")
        elif(healthcode in HC_healthcodes):
            output_file_path.append("HC/")
        else:
            output_file_path.append("Undisclosed/")

        output_file_path.append(f"{healthcode}/")
        Path("".join(output_file_path)).mkdir(parents=True, exist_ok=True)

        temp_filename = path.split("/")[-1][:-3].split('.m4a-')[-1] + 'm4a'
        output_file_path.append(temp_filename)
        output_file_path = "".join(output_file_path)

        Path(path).rename(output_file_path)

        # Convert m4a file to wav and delete m4a file
        convert_m4a_to_wav(output_file_path)
        os.remove(output_file_path)

    end_timer_and_print(event_name)

def download_MDVR_KCL_dataset():

    if(Path("data/dataset/MDVR_KCL").exists()):
        log("MDVR_KCL folder found - skipping redownload")
        return

    zipurl = 'https://zenodo.org/record/2867216/files/26_29_09_2017_KCL.zip?download=1'

    event_name = "Download MDVR_KCL Dataset"
    log(event_name)
    start_timer(event_name)
    try:
        with urlopen(zipurl) as zipresp:
            with ZipFile(BytesIO(zipresp.read())) as zfile:
                zfile.extractall('data/dataset/')
    except Exception as e:
        logging.error(f"Failed: {event_name}, {e}")
    end_timer_and_print(event_name)

    event_name = "Move MDVR_KCL Dataset"
    start_timer(event_name)
    source_dir = "data/dataset/26-29_09_2017_KCL/ReadText/"
    target_dir = "data/dataset/MDVR_KCL/"
    file_names = os.listdir(source_dir)
    
    try:
        Path(target_dir).mkdir(parents=True, exist_ok=True)
        for file_name in file_names:
            shutil.move(os.path.join(source_dir, file_name), os.path.join(target_dir, file_name))
    except Exception as e:
        logging.error(f"Failed: to move files {e}")
    shutil.rmtree("data/dataset/26-29_09_2017_KCL", onerror = lambda fn, filename, err: logging.error(err) )

    end_timer_and_print(event_name)

def download_italian_dataset():
    # # TODO: Requires IEEE DataPort Login - Consider pulling data from Aishat's repo
    # zipurl = 'https://ieee-dataport.s3.amazonaws.com/open/11738/Italian%20Parkinson%27s%20Voice%20and%20speech.zip?response-content-disposition=attachment%3B%20filename%3D%22Italian%20Parkinson%27s%20Voice%20and%20speech.zip%22&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAJOHYI4KJCE6Q7MIQ%2F20220621%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220621T160701Z&X-Amz-SignedHeaders=Host&X-Amz-Expires=86400&X-Amz-Signature=f483cabcc21de465093db71d950b886ae4b1dd958ea89679c0855e966440f932'

    # Use dataset stored in Aeesha-T's repo
    # https://github.com/aeesha-T/parkinsons_prediction_using_speech
    zipurl = "https://codeload.github.com/aeesha-T/parkinsons_prediction_using_speech/zip/refs/heads/master"
    

    if(Path("data/dataset/ItalianParkinsonSpeech").exists()):
        log("ItalianParkinsonSpeech folder found - skipping redownload")
        return

    event_name = "Download ItalianParkinsonSpeech Dataset"    
    log(event_name)
    start_timer(event_name)

    try:
        with urlopen(zipurl) as zipresp:
            with ZipFile(BytesIO(zipresp.read())) as zfile:
                zfile.extractall('data/dataset/')
    except Exception as e:
        logging.error(f"Failed: {event_name}, {e}")

    source_dir = "data/dataset/parkinsons_prediction_using_speech-master/dataset/ItalianParkinsonSpeech/"
    target_dir = "data/dataset/ItalianParkinsonSpeech/"
    file_names = os.listdir(source_dir)
    
    try:
        Path(target_dir).mkdir(parents=True, exist_ok=True)
        for file_name in file_names:
            shutil.move(os.path.join(source_dir, file_name), os.path.join(target_dir, file_name))
    except Exception as e:
        logging.error(f"Failed: to move files {e}")
    shutil.rmtree("data/dataset/parkinsons_prediction_using_speech-master", onerror = lambda fn, filename, err: logging.error(err) )

    end_timer_and_print(event_name)

def download_datasets():
    download_MDVR_KCL_dataset()
    download_italian_dataset()
    download_mPower_dataset()


if __name__ == "__main__":
    download_datasets()
