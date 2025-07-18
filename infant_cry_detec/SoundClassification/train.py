#!/usr/bin/python3
"""
To run this script, use the following command:
> python train_class_embeddings.py {hyperparameter_file}
"""

import os
import sys
from IPython.display import display
import librosa
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchaudio

# from torchaudio.transforms import Fade
import torchvision
import torch.nn.functional as F

# from torchaudio.models import ConvTasNet
from tools.confusion_matrix_fig import create_cm_fig
from hyperpyyaml import load_hyperpyyaml
from sklearn.metrics import confusion_matrix
from speechbrain.utils import hpopt as hp  # type: ignore

import speechbrain as sb  # type: ignore

# from speechbrain.utils.distributed import run_on_main # type: ignore

DEBUG = False
IS_IMG = False
TEST = True

class brain(sb.core.Brain):
    """Class for sound class embedding training"""

    def __init__(
        self,
        modules=None,
        opt_class=None,
        hparams=None,
        run_opts=None,
        checkpointer=None,
    ):
        super().__init__(modules, opt_class, hparams, run_opts, checkpointer)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"------Running on {self.device}!------")
        # 添加 Dropout 层
        self.dropout = torch.nn.Dropout(p=hparams["dropout_prob"])

    def compute_forward(self, batch, stage):
        """
        Computation pipeline.
        Data augmentation and environmental corruption are applied to the
        input sound.
        """
        batch = batch.to(self.device)
        wavs, lens = batch.sig

        # Add waveform augmentation if specified.
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment_train"):
            wavs, lens = self.hparams.wav_augment_train(wavs, lens)
        if stage != sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment_valid"):
            wavs, lens = self.hparams.wav_augment_valid(wavs, lens)

        if DEBUG:
            # 确保目标文件夹存在
            os.makedirs("myexp", exist_ok=True)
            for i in range(self.hparams.batch_size):
                chip = wavs[i]
                chip = chip.cpu()
                # 确保 wavs 是二维张量 (channels, samples)
                if chip.dim() == 1:
                    chip = chip.unsqueeze(0)

                # 保存音频文件
                output_path = os.path.join("myexp", f"example{i}.wav")
                torchaudio.save(output_path, chip, 16000)
                print(f"Saved wav{i}")

                # 终止程序
            sys.exit("程序已终止")

        feats = self.hparams.compute_features(wavs)

        if self.hparams.amp_to_db:
            Amp2db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
            feats = Amp2db(feats)

        if IS_IMG:
            # 在第二个维度上增加通道数
            feats = feats.unsqueeze(1).repeat(
                1, 3, 1, 1
            )  # 输出形状为 [16, 3, 128, 501]
            print(f"feats shape: {feats.shape}")

            # 定义图像预处理步骤
            tra = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize((224, 224), antialias=True),
                    torchvision.transforms.Normalize(
                        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
                    ),  # 像素归一化
                ]
            )
            feats = tra(feats)

        if DEBUG:
            os.makedirs("mypng", exist_ok=True)
            pngfeat = feats.cpu().numpy()
            for i in range(self.hparams.batch_size):
                # 创建图形对象，指定图形大小
                plt.figure(
                    figsize=(pngfeat.shape[-1] / 50, pngfeat.shape[-2] / 50), dpi=100
                )
                if pngfeat.ndim == 4:
                    librosa.display.specshow(pngfeat[i][1], sr=16000, fmax=8000)
                else:
                    librosa.display.specshow(pngfeat[i], sr=16000, fmax=8000)
                # librosa.display.specshow(librosa.power_to_db(feats[0], ref=np.max), sr=16000, fmax=8000)
                # plt.imshow(librosa.power_to_db(melspec[0]), origin="lower", aspect="auto", interpolation="nearest")
                plt.savefig(
                    f"mypng/spectrum{i}.png", bbox_inches="tight", pad_inches=0
                )  # 保存图像
                plt.close()  # 关闭图形窗口，释放内存

        # Embeddings + sound classifier
        embeddings = self.modules.embedding_model(feats)
        # 应用 Dropout
        # embeddings = self.dropout(embeddings)
        # print(f"embeddings shape: {embeddings.shape}")
        # embeddings = self.modules.classifier(embeddings)
        if embeddings.dim() != 3 and embeddings.size(1) != 1:
            classifier_outputs = embeddings.unsqueeze(1)
        else:
            classifier_outputs = embeddings

        return classifier_outputs, lens

    def compute_objectives(
        self, predictions, batch, stage
    ):  # predictions是forward的返回值
        """Computes the loss using class-id as label."""
        predictions, lens = predictions
        uttid = batch.id
        classid, _ = batch.class_string_encoded

        # Concatenate labels (due to data augmentation)
        if hasattr(self.hparams, "wav_augment"):
            classid = self.hparams.wav_augment.replicate_labels(classid)

        # 标签平滑操作
        if hasattr(self.hparams, "label_smoothing"):
            num_classes = predictions.size(-1)
            smooth_labels = (1 - self.hparams.label_smoothing) * F.one_hot(
                classid, num_classes
            ) + self.hparams.label_smoothing / num_classes
            smooth_labels = smooth_labels.float()
            loss = F.cross_entropy(predictions, smooth_labels, reduction="none")
            loss = (loss * lens).sum() / lens.sum()
        else:
            loss = self.hparams.compute_cost(predictions, classid, lens)

        if stage == sb.Stage.TRAIN and hasattr(
            self.hparams.lr_annealing, "on_batch_end"
        ):
            self.hparams.lr_annealing.on_batch_end(self.optimizer)

        # Append this batch of losses to the loss metric for easy
        self.loss_metric.append(uttid, predictions, classid, lens, reduction="batch")

        # Confusion matrices
        if stage != sb.Stage.TRAIN:
            y_true = classid.cpu().detach().numpy().squeeze(-1)
            y_pred = predictions.cpu().detach().numpy().argmax(-1).squeeze(-1)

        if stage == sb.Stage.VALID:
            my_confusion_matrix = confusion_matrix(
                y_true,
                y_pred,
                labels=sorted(self.hparams.label_encoder.ind2lab.keys()),
            )
            self.valid_confusion_matrix += my_confusion_matrix

        if stage == sb.Stage.TEST:
            my_confusion_matrix = confusion_matrix(
                y_true,
                y_pred,
                labels=sorted(self.hparams.label_encoder.ind2lab.keys()),
            )
            self.test_confusion_matrix += my_confusion_matrix

        # Compute Accuracy using MetricStats
        self.acc_metric.append(uttid, predict=predictions, target=classid, lengths=lens)

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(uttid, predictions, classid, lens)

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """
        # Set up statistics trackers for this stage
        torch.cuda.empty_cache()
        self.loss_metric = sb.utils.metric_stats.MetricStats(
            metric=sb.nnet.losses.nll_loss
        )

        # Compute Accuracy using MetricStats
        # Define function taking (prediction, target, length) for eval
        def accuracy_value(predict, target, lengths):
            """Computes Accuracy"""
            nbr_correct, nbr_total = sb.utils.Accuracy.Accuracy(
                predict, target, lengths
            )
            acc = torch.tensor([nbr_correct / nbr_total])
            return acc

        self.acc_metric = sb.utils.metric_stats.MetricStats(
            metric=accuracy_value, n_jobs=1
        )

        # Confusion matrices
        if stage == sb.Stage.VALID:
            self.valid_confusion_matrix = np.zeros(
                shape=(self.hparams.out_n_neurons, self.hparams.out_n_neurons),
                dtype=int,
            )
        if stage == sb.Stage.TEST:
            self.test_confusion_matrix = np.zeros(
                shape=(self.hparams.out_n_neurons, self.hparams.out_n_neurons),
                dtype=int,
            )

        # Set up evaluation-only statistics trackers
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """
        # Compute/store important stats
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
            self.train_stats = {
                "loss": self.train_loss,
                "acc": self.acc_metric.summarize(
                    "average"
                ),  # "acc": self.train_acc_metric.summarize(),
            }
        # Summarize Valid statistics from the stage for record-keeping.
        elif stage == sb.Stage.VALID:
            valid_stats = {
                "loss": stage_loss,
                "acc": self.acc_metric.summarize(
                    "average"
                ),  # "acc": self.valid_acc_metric.summarize(),
                "error": self.error_metrics.summarize("average"),
            }
            # hp.report_result(self.error_metrics.summarize("average"))
        # Summarize Test statistics from the stage for record-keeping.
        else:
            test_stats = {
                "loss": stage_loss,
                "acc": self.acc_metric.summarize(
                    "average"
                ),  # "acc": self.test_acc_metric.summarize(),
                "error": self.error_metrics.summarize("average"),
            }

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            # torch.Tensorboard logging
            if self.hparams.use_tensorboard:
                self.hparams.tensorboard_train_logger.log_stats(
                    stats_meta={"Epoch": epoch},
                    train_stats=self.train_stats,
                    valid_stats=valid_stats,
                )
                # Log confusion matrix fig to tensorboard
                cm_fig = create_cm_fig(
                    self.valid_confusion_matrix,
                    display_labels=list(self.hparams.label_encoder.ind2lab.values()),
                )
                self.hparams.tensorboard_train_logger.writer.add_figure(
                    "Validation Confusion Matrix", cm_fig, epoch
                )  # TODO use global_step from writer

            # Per class accuracy from Validation confusion matrix
            per_class_acc_arr = np.diag(self.valid_confusion_matrix) / np.sum(
                self.valid_confusion_matrix, axis=1
            )
            per_class_acc_arr_str = "\n" + "\n".join(
                "{:}: {:.3f}".format(
                    self.hparams.label_encoder.decode_ndim(class_id), class_acc
                )
                for class_id, class_acc in enumerate(per_class_acc_arr)
            )

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=valid_stats,
            )
            # Save the current checkpoint and delete previous checkpoints,
            self.checkpointer.save_and_keep_only(
                meta=valid_stats, min_keys=["error"], num_to_keep=3
            )

        # We also write statistics about test data to stdout and to the logfile.
        if stage == sb.Stage.TEST:
            # Per class accuracy from Test confusion matrix
            per_class_acc_arr = np.diag(self.test_confusion_matrix) / np.sum(
                self.test_confusion_matrix, axis=1
            )
            per_class_acc_arr_str = "\n" + "\n".join(
                "{:}: {:.3f}".format(class_id, class_acc)
                for class_id, class_acc in enumerate(per_class_acc_arr)
            )

            self.hparams.train_logger.log_stats(
                {
                    "Epoch loaded": self.hparams.epoch_counter.current,
                    "\n Per Class Accuracy": per_class_acc_arr_str,
                    "\n Confusion Matrix": "\n{:}\n".format(self.test_confusion_matrix),
                },
                test_stats=test_stats,
            )


def dataio_prep(hparams):
    "Creates the datasets and their data processing pipelines."

    data_audio_folder = hparams["audio_data_folder"]
    config_sample_rate = hparams["sample_rate"]
    label_encoder = sb.dataio.encoder.CategoricalEncoder()
    label_encoder.expect_len(hparams["out_n_neurons"])
    # TODO  use SB implementation but need to make sure it give the same results as PyTorch
    # resampler = sb.processing.speech_augmentation.Resample(orig_freq=latest_file_sr, new_freq=config_sample_rate)
    hparams["resampler"] = torchaudio.transforms.Resample(new_freq=config_sample_rate)

    # 2. Define audio pipeline:
    # @sb.utils.data_pipeline.takes("wav", "fold")
    @sb.utils.data_pipeline.takes("wav", "class_string")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, class_string):
        """Load the signal, and pass it and its length to the corruption class.
        This is done on the CPU in the `collate_fn`."""

        wave_file = data_audio_folder + "/{0:}/{1:}".format(class_string, wav)

        sig, read_sr = torchaudio.load(wave_file)

        # If multi-channels, downmix it to a mono channel
        sig = torch.squeeze(sig)
        if len(sig.shape) > 1:
            sig = torch.mean(sig, dim=0)

        # Convert sample rate to required config_sample_rate
        if read_sr != config_sample_rate:
            # Re-initialize sampler if source file sample rate changed compared to last file
            if read_sr != hparams["resampler"].orig_freq:
                hparams["resampler"] = torchaudio.transforms.Resample(
                    orig_freq=read_sr, new_freq=config_sample_rate
                )
            # Resample audio
            sig = hparams["resampler"].forward(sig)

        return sig

    # 3. Define label pipeline:
    @sb.utils.data_pipeline.takes("class_string")
    @sb.utils.data_pipeline.provides("class_string", "class_string_encoded")
    def label_pipeline(class_string):
        yield class_string
        class_string_encoded = label_encoder.encode_label_torch(class_string)
        yield class_string_encoded

    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    datasets = {}
    data_info = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
        # "test": hparams["test_annotation"],
    }
    for dataset in data_info:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline, label_pipeline],
            output_keys=["id", "sig", "class_string_encoded"],
        )

    # Load or compute the label encoder (with multi-GPU DDP support)
    # Please, take a look into the lab_enc_file to see the label to index
    # mapping.
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[datasets["train"]],
        output_key="class_string",
    )

    return datasets, label_encoder


if __name__ == "__main__":
    # This flag enables the inbuilt cudnn auto-tuner
    torch.backends.cudnn.benchmark = True

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # torch.Tensorboard logging
    if hparams["use_tensorboard"]:
        from speechbrain.utils.train_logger import TensorboardLogger  # type: ignore

        hparams["tensorboard_train_logger"] = TensorboardLogger(
            hparams["tensorboard_logs_folder"]
        )

    # 从csv读取所有的train的标签数据
    train_metadata = pd.read_csv(
        f"{hparams['data_folder']}/metadata/{hparams['csv_name']}",
        dtype={"class_string": str, "ID": str},
    )
    # 过滤fold列数据为1, 2, 3和4的数据
    train_metadata = train_metadata.query(f"fold in {hparams['train_fold_nums']}")
    # train_metadata = metadata.loc[metadata["split"] == "train"].copy()
    display(
        train_metadata.head()
        .style.set_caption("train_metadata")
        .set_table_styles([{"selector": "caption", "props": [("font-size", "20px")]}])
    )
    display(train_metadata.describe())
    train_metadata.set_index("Name").to_json(
        hparams["train_annotation"], orient="index"
    )

    # 从csv读取所有的val的标签数据
    val_metadata = pd.read_csv(
        f"{hparams['data_folder']}/metadata/{hparams['csv_name']}",
        dtype={"class_string": str, "ID": str},
    )
    val_metadata = val_metadata.query(f"fold in {hparams['valid_fold_nums']}")
    # train_metadata = metadata.loc[metadata["split"] == "train"].copy()
    display(
        val_metadata.head()
        .style.set_caption("valid_metadata")
        .set_table_styles([{"selector": "caption", "props": [("font-size", "20px")]}])
    )
    display(val_metadata.describe())
    val_metadata.set_index("Name").to_json(hparams["valid_annotation"], orient="index")

    # Dataset IO prep: creating Dataset objects and proper encodings for phones
    datasets, label_encoder = dataio_prep(hparams)
    hparams["label_encoder"] = label_encoder

    class_labels = list(label_encoder.ind2lab.values())
    print("Class Labels:", class_labels)

    # with hp.hyperparameter_optimization(objective_key="error") as hp_ctx: # <-- Initialize the context

    pipe = brain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # if a pretrained model is specified, load it
    if "pretrained_embedding_model" in hparams and hparams["use_pretrained_model"]:
        sb.utils.distributed.run_on_main(
            hparams["pretrained_embedding_model"].collect_files
        )
        hparams["pretrained_embedding_model"].load_collected()
        print("Pretrained model loaded")

    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.
    # with torch.autograd.detect_anomaly():

    if not TEST:
        pipe.fit(
            epoch_counter=pipe.hparams.epoch_counter,
            train_set=datasets["train"],
            valid_set=datasets["valid"],
            train_loader_kwargs=hparams["dataloader_options"],
            valid_loader_kwargs=hparams["dataloader_options"],
        )
    else:
        # Load the best checkpoint for evaluation
        test_stats = pipe.evaluate(
            test_set=datasets["valid"],
            min_key="error",
            progressbar=True,
            test_loader_kwargs=hparams["dataloader_options"],
        )
        
