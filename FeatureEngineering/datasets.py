USE_BOOT_DISK_FOR_STORING_SYNTHETICS = True

prefix = ""
if USE_BOOT_DISK_FOR_STORING_SYNTHETICS:
    from pathlib import Path
    prefix = f"{Path.home()}/"

class DatasetInfo:
    def __init__(self, name, source_HC, source_PD, sink_HC = "", sink_PD = "", create_dataset=False, method = "none", duration = 0) -> None:
        
        methods = ["silence", "chunk"]
        if(create_dataset and (method not in methods)):
            raise Exception(f"Cannot create dataset with invalid method {method}")
        if(create_dataset and ((len(sink_HC) == 0) or (len(sink_PD) == 0))):
            raise Exception(f"Output path for dataset creation not specified")

        self.name = name
        self.source_HC = source_HC
        self.source_PD = source_PD
        self.sink_HC = prefix + sink_HC
        self.sink_PD = prefix + sink_PD
        self.create_dataset = create_dataset
        self.method = method
        self.duration = duration



    def __repr__(self) -> str:
        return f"name = {self.name}, source_HC = {self.source_HC}, source_PD = {self.source_PD}, sink_HC = {self.sink_HC}, sink_PD = {self.sink_PD}, create_dataset = {self.create_dataset}, method = {self.method}, duration = {self.duration}"

mPower_configs = [
    DatasetInfo(
        name="mPower", 
        source_HC=r"data/dataset/mPower/HC/*/*.wav", 
        source_PD=r"data/dataset/mPower/PD/*/*.wav",  
    ),
    DatasetInfo(
        name="mPower_min_silence_500ms",
        source_HC=r"data/dataset/mPower/HC/*/*.wav",
        source_PD=r"data/dataset/mPower/PD/*/*.wav",
        sink_HC=r"data/dataset/mPower_min_silence_500ms/HC",
        sink_PD=r"data/dataset/mPower_min_silence_500ms/PD",
        create_dataset=True,
        method="silence",
        duration=500
    ),
    DatasetInfo(
        name="mPower_min_silence_1000ms",
        source_HC=r"data/dataset/mPower/HC/*/*.wav",
        source_PD=r"data/dataset/mPower/PD/*/*.wav",
        sink_HC=r"data/dataset/mPower_min_silence_1000ms/HC",
        sink_PD=r"data/dataset/mPower_min_silence_1000ms/PD",
        create_dataset=True,
        method="silence",
        duration=1000
    ),
    DatasetInfo(
        name="mPower_min_silence_2000ms",
        source_HC=r"data/dataset/mPower/HC/*/*.wav",
        source_PD=r"data/dataset/mPower/PD/*/*.wav",
        sink_HC=r"data/dataset/mPower_min_silence_2000ms/HC",
        sink_PD=r"data/dataset/mPower_min_silence_2000ms/PD",
        create_dataset=True,
        method="silence",
        duration=2000
    ),
    DatasetInfo(
        name="mPower_chunk_500ms",
        source_HC=r"data/dataset/mPower/HC/*/*.wav",
        source_PD=r"data/dataset/mPower/PD/*/*.wav",
        sink_HC=r"data/dataset/mPower_chunk_500ms/HC",
        sink_PD=r"data/dataset/mPower_chunk_500ms/PD",
        create_dataset=True,
        method="chunk",
        duration=500
    ),
    DatasetInfo(
        name="mPower_chunk_1000ms",
        source_HC=r"data/dataset/mPower/HC/*/*.wav",
        source_PD=r"data/dataset/mPower/PD/*/*.wav",
        sink_HC=r"data/dataset/mPower_chunk_1000ms/HC",
        sink_PD=r"data/dataset/mPower_chunk_1000ms/PD",
        create_dataset=True,
        method="chunk",
        duration=1000
    ),
    DatasetInfo(
        name="mPower_chunk_3000ms",
        source_HC=r"data/dataset/mPower/HC/*/*.wav",
        source_PD=r"data/dataset/mPower/PD/*/*.wav",
        sink_HC=r"data/dataset/mPower_chunk_3000ms/HC",
        sink_PD=r"data/dataset/mPower_chunk_3000ms/PD",
        create_dataset=True,
        method="chunk",
        duration=3000
    ),
    DatasetInfo(
        name="mPower_chunk_5000ms",
        source_HC=r"data/dataset/mPower/HC/*/*.wav",
        source_PD=r"data/dataset/mPower/PD/*/*.wav",
        sink_HC=r"data/dataset/mPower_chunk_5000ms/HC",
        sink_PD=r"data/dataset/mPower_chunk_5000ms/PD",
        create_dataset=True,
        method="chunk",
        duration=5000
    ),
]


ItalianParkinsonSpeech_configs = [
    DatasetInfo(
        name="ItalianParkinsonSpeech", 
        source_HC=r"data/dataset/ItalianParkinsonSpeech/EHC/*/*.wav",
        source_PD=r"data/dataset/ItalianParkinsonSpeech/PD/*/*.wav",  
    ),
    DatasetInfo(
        name="ItalianParkinsonSpeech_min_silence_500ms",
        source_HC=r"data/dataset/ItalianParkinsonSpeech/EHC/*/*.wav",
        source_PD=r"data/dataset/ItalianParkinsonSpeech/PD/*/*.wav",
        sink_HC=r"data/dataset/ItalianParkinsonSpeech_min_silence_500ms/HC",
        sink_PD=r"data/dataset/ItalianParkinsonSpeech_min_silence_500ms/PD",
        create_dataset=True,
        method="silence",
        duration=500
    ),
    DatasetInfo(
        name="ItalianParkinsonSpeech_min_silence_1000ms",
        source_HC=r"data/dataset/ItalianParkinsonSpeech/EHC/*/*.wav",
        source_PD=r"data/dataset/ItalianParkinsonSpeech/PD/*/*.wav",
        sink_HC=r"data/dataset/ItalianParkinsonSpeech_min_silence_1000ms/HC",
        sink_PD=r"data/dataset/ItalianParkinsonSpeech_min_silence_1000ms/PD",
        create_dataset=True,
        method="silence",
        duration=1000
    ),
    DatasetInfo(
        name="ItalianParkinsonSpeech_min_silence_2000ms",
        source_HC=r"data/dataset/ItalianParkinsonSpeech/EHC/*/*.wav",
        source_PD=r"data/dataset/ItalianParkinsonSpeech/PD/*/*.wav",
        sink_HC=r"data/dataset/ItalianParkinsonSpeech_min_silence_2000ms/HC",
        sink_PD=r"data/dataset/ItalianParkinsonSpeech_min_silence_2000ms/PD",
        create_dataset=True,
        method="silence",
        duration=2000
    ),
    DatasetInfo(
        name="ItalianParkinsonSpeech_chunk_500ms",
        source_HC=r"data/dataset/ItalianParkinsonSpeech/EHC/*/*.wav",
        source_PD=r"data/dataset/ItalianParkinsonSpeech/PD/*/*.wav",
        sink_HC=r"data/dataset/ItalianParkinsonSpeech_chunk_500ms/HC",
        sink_PD=r"data/dataset/ItalianParkinsonSpeech_chunk_500ms/PD",
        create_dataset=True,
        method="chunk",
        duration=500
    ),
    DatasetInfo(
        name="ItalianParkinsonSpeech_chunk_1000ms",
        source_HC=r"data/dataset/ItalianParkinsonSpeech/EHC/*/*.wav",
        source_PD=r"data/dataset/ItalianParkinsonSpeech/PD/*/*.wav",
        sink_HC=r"data/dataset/ItalianParkinsonSpeech_chunk_1000ms/HC",
        sink_PD=r"data/dataset/ItalianParkinsonSpeech_chunk_1000ms/PD",
        create_dataset=True,
        method="chunk",
        duration=1000
    ),
    DatasetInfo(
        name="ItalianParkinsonSpeech_chunk_3000ms",
        source_HC=r"data/dataset/ItalianParkinsonSpeech/EHC/*/*.wav",
        source_PD=r"data/dataset/ItalianParkinsonSpeech/PD/*/*.wav",
        sink_HC=r"data/dataset/ItalianParkinsonSpeech_chunk_3000ms/HC",
        sink_PD=r"data/dataset/ItalianParkinsonSpeech_chunk_3000ms/PD",
        create_dataset=True,
        method="chunk",
        duration=3000
    ),
    DatasetInfo(
        name="ItalianParkinsonSpeech_chunk_5000ms",
        source_HC=r"data/dataset/ItalianParkinsonSpeech/EHC/*/*.wav",
        source_PD=r"data/dataset/ItalianParkinsonSpeech/PD/*/*.wav",
        sink_HC=r"data/dataset/ItalianParkinsonSpeech_chunk_5000ms/HC",
        sink_PD=r"data/dataset/ItalianParkinsonSpeech_chunk_5000ms/PD",
        create_dataset=True,
        method="chunk",
        duration=5000
    ),
]


MDVR_KCL_configs = [
    DatasetInfo(
        name="MDVR_KCL",
        source_HC=r"data/dataset/MDVR_KCL/HC/*.wav",
        source_PD=r"data/dataset/MDVR_KCL/PD/*.wav",
    ),
    DatasetInfo(
        name="MDVR_KCL_min_silence_500ms",
        source_HC=r"data/dataset/MDVR_KCL/HC/*.wav",
        source_PD=r"data/dataset/MDVR_KCL/PD/*.wav",
        sink_HC=r"data/dataset/MDVR_KCL_min_silence_500ms/HC",
        sink_PD=r"data/dataset/MDVR_KCL_min_silence_500ms/PD",
        create_dataset=True,
        method="silence",
        duration=500
    ),
    DatasetInfo(
        name="MDVR_KCL_min_silence_1000ms",
        source_HC=r"data/dataset/MDVR_KCL/HC/*.wav",
        source_PD=r"data/dataset/MDVR_KCL/PD/*.wav",
        sink_HC=r"data/dataset/MDVR_KCL_min_silence_1000ms/HC",
        sink_PD=r"data/dataset/MDVR_KCL_min_silence_1000ms/PD",
        create_dataset=True,
        method="silence",
        duration=1000
    ),
    DatasetInfo(
        name="MDVR_KCL_min_silence_2000ms",
        source_HC=r"data/dataset/MDVR_KCL/HC/*.wav",
        source_PD=r"data/dataset/MDVR_KCL/PD/*.wav",
        sink_HC=r"data/dataset/MDVR_KCL_min_silence_2000ms/HC",
        sink_PD=r"data/dataset/MDVR_KCL_min_silence_2000ms/PD",
        create_dataset=True,
        method="silence",
        duration=2000
    ),
    DatasetInfo(
        name="MDVR_KCL_chunk_500ms",
        source_HC=r"data/dataset/MDVR_KCL/HC/*.wav",
        source_PD=r"data/dataset/MDVR_KCL/PD/*.wav",
        sink_HC=r"data/dataset/MDVR_KCL_chunk_500ms/HC",
        sink_PD=r"data/dataset/MDVR_KCL_chunk_500ms/PD",
        create_dataset=True,
        method="chunk",
        duration=500
    ),
    DatasetInfo(
        name="MDVR_KCL_chunk_1000ms",
        source_HC=r"data/dataset/MDVR_KCL/HC/*.wav",
        source_PD=r"data/dataset/MDVR_KCL/PD/*.wav",
        sink_HC=r"data/dataset/MDVR_KCL_chunk_1000ms/HC",
        sink_PD=r"data/dataset/MDVR_KCL_chunk_1000ms/PD",
        create_dataset=True,
        method="chunk",
        duration=1000
    ),
    DatasetInfo(
        name="MDVR_KCL_chunk_3000ms",
        source_HC=r"data/dataset/MDVR_KCL/HC/*.wav",
        source_PD=r"data/dataset/MDVR_KCL/PD/*.wav",
        sink_HC=r"data/dataset/MDVR_KCL_chunk_3000ms/HC",
        sink_PD=r"data/dataset/MDVR_KCL_chunk_3000ms/PD",
        create_dataset=True,
        method="chunk",
        duration=3000
    ),
    DatasetInfo(
        name="MDVR_KCL_chunk_5000ms",
        source_HC=r"data/dataset/MDVR_KCL/HC/*.wav",
        source_PD=r"data/dataset/MDVR_KCL/PD/*.wav",
        sink_HC=r"data/dataset/MDVR_KCL_chunk_5000ms/HC",
        sink_PD=r"data/dataset/MDVR_KCL_chunk_5000ms/PD",
        create_dataset=True,
        method="chunk",
        duration=5000
    ),
]


datasets = [*MDVR_KCL_configs, *ItalianParkinsonSpeech_configs, *mPower_configs]