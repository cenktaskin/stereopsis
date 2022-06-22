from tensorflow.core.util.event_pb2 import Event
import tensorflow as tf
import os
import glob
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

tag_dict = {"Loss/train": "RunningLoss/iteration",
            "Avg Losses per Epoch": "Loss/epoch",
            "Learning Curve [EPE/Epoch]": "Error/epoch",
            "Accuracy/epoch": "Error/epoch",
            "RunningLoss/iteration": "RunningLoss/iteration",
            "Running loss/iteration": "RunningLoss/iteration",
            "Loss/epoch": "Loss/epoch"}

# Make a record writer
logs_dir = Path("/home/cenkt/git/stereopsis/data/logs/")
path = logs_dir.joinpath("dispnet-202206210420-phisrv-PretrainedComparisonFalse")
event_files = list(glob.glob(f"{path}/**/event*", recursive=True))

already_seen = []
for p in event_files:
    new_file = p + ".new"
    with tf.io.TFRecordWriter(new_file) as writer:
        # Iterate event records
        for rec in tf.data.TFRecordDataset([p]):
            # Read event
            ev = Event()
            ev.MergeFromString(rec.numpy())
            # Check if it is a summary
            if ev.summary:
                # Iterate summary values
                for v in ev.summary.value:
                    # Check if the tag should be renamed
                    if v.tag not in tag_dict:
                        new_tag = input(f"Provide new tag for {v.tag}:")
                        tag_dict[v.tag] = new_tag

                    if v.tag in tag_dict:
                        v.tag = tag_dict[v.tag]
            writer.write(ev.SerializeToString())
    os.rename(new_file, p)
