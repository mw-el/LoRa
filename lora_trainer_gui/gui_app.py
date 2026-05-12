from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QStatusBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .config import DefaultConfig, default_config, load_last_settings, save_settings
from .training import train_lora


def _timestamp() -> str:
    return datetime.now().strftime("[%H:%M:%S]")


class TrainingWorker(QObject):
    progress = pyqtSignal(str)
    status = pyqtSignal(str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, args: dict) -> None:
        super().__init__()
        self.args = args
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def run(self) -> None:
        try:
            result = train_lora(
                **self.args,
                progress_callback=self.progress.emit,
                status_callback=self.status.emit,
                stop_flag=lambda: self._stop_requested,
            )
            if self._stop_requested:
                self.cancelled.emit()
            else:
                self.finished.emit(str(result))
        except Exception as exc:  # pragma: no cover - handled in GUI layer
            self.error.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Style LoRA Trainer")
        self.resize(900, 700)
        self.config: DefaultConfig = default_config()
        self.default_sdxl_model = "stabilityai/stable-diffusion-xl-base-1.0"
        self.default_sd15_model = "/home/matthias/_AA_ComfyUI/models/checkpoints/realisticVisionV60B1_v51HyperVAE.safetensors"
        self.worker_thread: Optional[QThread] = None
        self.worker: Optional[TrainingWorker] = None
        self._build_ui()
        self._load_settings()

    def _build_ui(self) -> None:
        central = QWidget()
        main_layout = QVBoxLayout()

        main_layout.addWidget(self._build_model_section())
        main_layout.addWidget(self._build_dataset_section())
        main_layout.addWidget(self._build_params_section())
        main_layout.addWidget(self._build_control_section())

        central.setLayout(main_layout)
        self.setCentralWidget(central)
        self._status_label = QLabel("Idle")
        status_bar = QStatusBar()
        status_bar.addPermanentWidget(self._status_label)
        self.setStatusBar(status_bar)

    def _build_model_section(self) -> QGroupBox:
        box = QGroupBox("Model configuration")
        layout = QGridLayout()

        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(["SDXL", "SD 1.5"])
        self.model_type_combo.currentTextChanged.connect(self._on_model_type_changed)

        self.base_model_edit = QLineEdit()
        self.base_model_browse = QPushButton("Browse…")
        self.base_model_browse.clicked.connect(self._browse_base_model)

        self.output_dir_edit = QLineEdit()
        self.output_dir_browse = QPushButton("Browse…")
        self.output_dir_browse.clicked.connect(self._browse_output_dir)

        layout.addWidget(QLabel("Base model type"), 0, 0)
        layout.addWidget(self.model_type_combo, 0, 1, 1, 2)
        layout.addWidget(QLabel("Base model path or HF ID"), 1, 0)
        layout.addWidget(self.base_model_edit, 1, 1)
        layout.addWidget(self.base_model_browse, 1, 2)
        layout.addWidget(QLabel("Output folder"), 2, 0)
        layout.addWidget(self.output_dir_edit, 2, 1)
        layout.addWidget(self.output_dir_browse, 2, 2)

        box.setLayout(layout)
        return box

    def _build_dataset_section(self) -> QGroupBox:
        box = QGroupBox("Dataset")
        layout = QGridLayout()

        self.image_dir_edit = QLineEdit()
        browse_images = QPushButton("Browse…")
        browse_images.clicked.connect(self._browse_image_dir)

        self.use_captions_checkbox = QCheckBox("Use captions from .txt next to images")
        self.style_token_edit = QLineEdit()

        layout.addWidget(QLabel("Image folder"), 0, 0)
        layout.addWidget(self.image_dir_edit, 0, 1)
        layout.addWidget(browse_images, 0, 2)
        layout.addWidget(self.use_captions_checkbox, 1, 0, 1, 3)
        layout.addWidget(QLabel("Generic style token"), 2, 0)
        layout.addWidget(self.style_token_edit, 2, 1, 1, 2)

        box.setLayout(layout)
        return box

    def _build_params_section(self) -> QGroupBox:
        box = QGroupBox("Training parameters")
        layout = QGridLayout()

        self.resolution_spin = QSpinBox()
        self.resolution_spin.setRange(64, 2048)
        self.resolution_spin.setValue(self.config.resolution)

        self.rank_spin = QSpinBox()
        self.rank_spin.setRange(1, 128)
        self.rank_spin.setValue(self.config.lora_rank)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(6)
        self.lr_spin.setRange(1e-6, 1.0)
        self.lr_spin.setSingleStep(1e-5)
        self.lr_spin.setValue(self.config.learning_rate)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(self.config.num_epochs)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 32)
        self.batch_spin.setValue(self.config.batch_size)

        self.grad_ckpt_check = QCheckBox("Use gradient checkpointing")
        self.grad_ckpt_check.setChecked(self.config.gradient_checkpointing)

        self.fp16_check = QCheckBox("Enable mixed precision (fp16)")
        self.fp16_check.setChecked(self.config.mixed_precision)

        self.max_steps_spin = QSpinBox()
        self.max_steps_spin.setRange(0, 1000000)
        self.max_steps_spin.setValue(0)
        self.max_steps_spin.setSpecialValueText("Auto")

        layout.addWidget(QLabel("Resolution"), 0, 0)
        layout.addWidget(self.resolution_spin, 0, 1)
        layout.addWidget(QLabel("LoRA rank"), 0, 2)
        layout.addWidget(self.rank_spin, 0, 3)

        layout.addWidget(QLabel("Learning rate"), 1, 0)
        layout.addWidget(self.lr_spin, 1, 1)
        layout.addWidget(QLabel("Epochs"), 1, 2)
        layout.addWidget(self.epochs_spin, 1, 3)

        layout.addWidget(QLabel("Batch size"), 2, 0)
        layout.addWidget(self.batch_spin, 2, 1)
        layout.addWidget(QLabel("Max steps"), 2, 2)
        layout.addWidget(self.max_steps_spin, 2, 3)

        layout.addWidget(self.grad_ckpt_check, 3, 0, 1, 2)
        layout.addWidget(self.fp16_check, 3, 2, 1, 2)

        box.setLayout(layout)
        return box

    def _build_control_section(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout()

        buttons_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Training")
        self.start_button.clicked.connect(self._start_training)
        self.cancel_button = QPushButton("Cancel Training")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel_training)
        buttons_layout.addWidget(self.start_button)
        buttons_layout.addWidget(self.cancel_button)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMinimumHeight(200)

        layout.addLayout(buttons_layout)
        layout.addWidget(QLabel("Log"))
        layout.addWidget(self.log_output)
        container.setLayout(layout)
        return container

    def _on_model_type_changed(self, text: str) -> None:
        if text.startswith("SD 1.5"):
            self.resolution_spin.setValue(768)
            # Prefer known local SD1.5 checkpoint; keep user override if already set.
            if not self.base_model_edit.text().strip() or self.base_model_edit.text().strip() == self.default_sdxl_model:
                self.base_model_edit.setText(self.default_sd15_model)
        else:
            # SDXL high-quality overnight default; adjust down if OOM.
            self.resolution_spin.setValue(896)
            # Only set a default if the field is empty or still pointing to the SD1.5 default.
            current = self.base_model_edit.text().strip()
            if not current or current == self.default_sd15_model:
                self.base_model_edit.setText(self.default_sdxl_model)

    def _browse_base_model(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Select base model checkpoint")
        if file_path:
            self.base_model_edit.setText(file_path)

    def _browse_output_dir(self) -> None:
        dir_path = QFileDialog.getExistingDirectory(self, "Select output folder")
        if dir_path:
            self.output_dir_edit.setText(dir_path)

    def _browse_image_dir(self) -> None:
        dir_path = QFileDialog.getExistingDirectory(self, "Select image folder")
        if dir_path:
            self.image_dir_edit.setText(dir_path)

    def _append_log(self, message: str) -> None:
        self.log_output.append(f"{_timestamp()} {message}")

    def _set_status(self, message: str) -> None:
        self._status_label.setText(message)

    def _validate_inputs(self) -> Optional[str]:
        base_model = self.base_model_edit.text().strip()
        image_dir = Path(self.image_dir_edit.text().strip())
        output_dir = Path(self.output_dir_edit.text().strip())
        style_token = self.style_token_edit.text().strip()

        if not base_model:
            return "Base model path or Hugging Face model ID is required."
        bm_path = Path(base_model)
        if bm_path.exists() and bm_path.is_file():
            if bm_path.suffix.lower() not in {".safetensors", ".ckpt"}:
                return "Base model file must be a .safetensors or .ckpt checkpoint."
            if bm_path.stat().st_size < 1_000_000:
                return "Base model file looks empty. Please verify the checkpoint."
        elif bm_path.exists() and bm_path.is_dir():
            # Allow a directory if it contains at least one checkpoint.
            candidates = list(bm_path.glob("*.safetensors")) + list(bm_path.glob("*.ckpt"))
            if not candidates:
                return "Base model directory has no .safetensors/.ckpt checkpoint."
        if not image_dir.exists() or not image_dir.is_dir():
            return "Image folder does not exist."
        image_files = [p for p in image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}]
        if not image_files:
            return "Image folder is empty or has no supported images."
        if not output_dir.exists():
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                return "Output folder cannot be created."
        if not style_token:
            return "Style token is required."
        return None

    def _collect_args(self) -> dict:
        base_model_type = self.model_type_combo.currentText()
        model_type_internal = "sd15" if "1.5" in base_model_type else "sdxl"
        max_steps = self.max_steps_spin.value()
        return {
            "base_model": self.base_model_edit.text().strip(),
            "model_type": model_type_internal,
            "image_dir": Path(self.image_dir_edit.text().strip()),
            "output_dir": Path(self.output_dir_edit.text().strip()),
            "style_token": self.style_token_edit.text().strip(),
            "use_captions": self.use_captions_checkbox.isChecked(),
            "resolution": self.resolution_spin.value(),
            "lora_rank": self.rank_spin.value(),
            "learning_rate": self.lr_spin.value(),
            "num_epochs": self.epochs_spin.value(),
            "batch_size": self.batch_spin.value(),
            "use_gradient_checkpointing": self.grad_ckpt_check.isChecked(),
            "use_fp16": self.fp16_check.isChecked(),
            "max_train_steps": max_steps if max_steps > 0 else None,
        }

    def _start_training(self) -> None:
        error = self._validate_inputs()
        if error:
            QMessageBox.warning(self, "Invalid configuration", error)
            self._append_log(error)
            return

        args = self._collect_args()
        self._persist_settings()

        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self._set_status("Training…")
        self._append_log("Starting training run.")

        self.worker = TrainingWorker(args)
        self.worker_thread = QThread()
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._append_log)
        self.worker.status.connect(self._set_status)
        self.worker.error.connect(self._on_worker_error)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.cancelled.connect(self._on_worker_cancelled)
        self.worker_thread.start()

    def _cancel_training(self) -> None:
        if self.worker:
            self.worker.request_stop()
            self._append_log("Cancellation requested.")
            self._set_status("Cancelling…")
        self.cancel_button.setEnabled(False)

    def _cleanup_worker(self) -> None:
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()
        self.worker_thread = None
        self.worker = None
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)

    def _on_worker_finished(self, result_path: str) -> None:
        self._append_log(f"Training finished. Saved to: {result_path}")
        self._set_status("Finished")
        QMessageBox.information(self, "Training finished", f"LoRA saved to:\n{result_path}")
        self._cleanup_worker()

    def _on_worker_error(self, message: str) -> None:
        self._append_log(f"Error: {message}")
        self._set_status("Error")
        QMessageBox.critical(self, "Training error", message)
        self._cleanup_worker()

    def _on_worker_cancelled(self) -> None:
        self._append_log("Training cancelled by user.")
        self._set_status("Cancelled")
        QMessageBox.information(self, "Cancelled", "Training was cancelled.")
        self._cleanup_worker()

    def _persist_settings(self) -> None:
        data = {
            "model_type": self.model_type_combo.currentText(),
            "resolution": self.resolution_spin.value(),
            "lora_rank": self.rank_spin.value(),
            "learning_rate": self.lr_spin.value(),
            "num_epochs": self.epochs_spin.value(),
            "batch_size": self.batch_spin.value(),
            "gradient_checkpointing": self.grad_ckpt_check.isChecked(),
            "mixed_precision": self.fp16_check.isChecked(),
            "max_train_steps": self.max_steps_spin.value(),
            "base_model": self.base_model_edit.text().strip(),
            "output_dir": self.output_dir_edit.text().strip(),
            "image_dir": self.image_dir_edit.text().strip(),
            "style_token": self.style_token_edit.text().strip(),
            "use_captions": self.use_captions_checkbox.isChecked(),
        }
        save_settings(data)

    def _load_settings(self) -> None:
        saved = load_last_settings()
        if saved:
            self.model_type_combo.setCurrentText(saved.get("model_type", self.config.model_type))
            self.resolution_spin.setValue(int(saved.get("resolution", self.config.resolution)))
            self.rank_spin.setValue(int(saved.get("lora_rank", self.config.lora_rank)))
            self.lr_spin.setValue(float(saved.get("learning_rate", self.config.learning_rate)))
            self.epochs_spin.setValue(int(saved.get("num_epochs", self.config.num_epochs)))
            self.batch_spin.setValue(int(saved.get("batch_size", self.config.batch_size)))
            self.grad_ckpt_check.setChecked(bool(saved.get("gradient_checkpointing", self.config.gradient_checkpointing)))
            self.fp16_check.setChecked(bool(saved.get("mixed_precision", self.config.mixed_precision)))
            self.max_steps_spin.setValue(int(saved.get("max_train_steps", 0) or 0))
            self.base_model_edit.setText(saved.get("base_model", ""))
            self.output_dir_edit.setText(saved.get("output_dir", ""))
            self.image_dir_edit.setText(saved.get("image_dir", ""))
            self.style_token_edit.setText(saved.get("style_token", ""))
            self.use_captions_checkbox.setChecked(bool(saved.get("use_captions", False)))
        else:
            self.model_type_combo.setCurrentText(self.config.model_type)
            if self.config.base_model:
                self.base_model_edit.setText(self.config.base_model)
            else:
                # Fallback to SD1.5 local default to avoid missing checkpoints.
                self.base_model_edit.setText(self.default_sd15_model)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._persist_settings()
        return super().closeEvent(event)


def run_app() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
