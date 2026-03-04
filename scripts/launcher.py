"""
launcher.py — DDSH Neural Edge Launcher (Proxima 2026 Edition)
Premium PyQt6 GUI with advanced styling, drop shadows, and modern UX.
"""

import sys
import subprocess
import os
import time
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QComboBox, QPushButton, QFrame, QGraphicsDropShadowEffect)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QFont

class DDSHLauncher(QWidget):
    def __init__(self):
        super().__init__()
        
        # Path Resolution 
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.root_dir = os.path.dirname(self.script_dir)
        self.venv_python = os.path.join(self.root_dir, "venv", "bin", "python")
        self.target_script = os.path.join(self.script_dir, "pose_integrate.py")
        
        self.initUI()

    def initUI(self):
        self.setWindowTitle('DDSH - Neural Edge Initialization')
        self.setFixedSize(550, 400)
        
        # Base Window Background (Deep Space Blue/Black)
        self.setStyleSheet("background-color: #0A0E17; font-family: 'Courier New', Courier, monospace;")
        
        # Main Layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(30, 30, 30, 30)
        
        # --- CENTRAL PANEL (Glass/Card Effect) ---
        self.panel = QFrame()
        self.panel.setStyleSheet("""
            QFrame {
                background-color: #121824;
                border-radius: 12px;
                border: 1px solid #1E293B;
            }
        """)
        panel_layout = QVBoxLayout(self.panel)
        panel_layout.setContentsMargins(30, 30, 30, 30)
        panel_layout.setSpacing(20)

        # --- LOGO SECTION ---
        # Using HTML to perfectly style the text and the red dot
        self.logo = QLabel("<span style='color: #00E5FF;'>DDSH</span> <span style='color: #FF0055;'>•</span>")
        self.logo.setStyleSheet("""
            QLabel { 
                font-size: 52px; 
                font-weight: bold; 
                font-family: 'Bank Gothic', Arial; 
                background: transparent;
                border: none;
            }
        """)
        self.logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        panel_layout.addWidget(self.logo)
        
        self.sub = QLabel("UNIFIED EDGE PIPELINE v3.0")
        self.sub.setStyleSheet("color: #64748B; font-size: 13px; font-weight: bold; letter-spacing: 2px; border: none; background: transparent;")
        self.sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        panel_layout.addWidget(self.sub)
        
        panel_layout.addSpacing(15)
        
        # --- CAMERA DROPDOWN ---
        self.cam_label = QLabel("SELECT OPTICAL SENSOR:")
        self.cam_label.setStyleSheet("color: #38BDF8; font-size: 12px; font-weight: bold; border: none; background: transparent;")
        panel_layout.addWidget(self.cam_label)
        
        self.combo = QComboBox()
        self.combo.addItems(["iPhone Continuity (Index 0)", "MacBook Camera (Index 1)", "External USB (Index 2)"])
        self.combo.setCurrentIndex(1)
        self.combo.setStyleSheet("""
            QComboBox { 
                background-color: #0F172A; 
                color: #E2E8F0; 
                font-size: 14px; 
                padding: 12px; 
                border: 1px solid #38BDF8; 
                border-radius: 6px;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView { 
                background-color: #0F172A; 
                color: #E2E8F0; 
                selection-background-color: #0284C7; 
            }
        """)
        panel_layout.addWidget(self.combo)
        
        panel_layout.addSpacing(10)
        
        # --- START BUTTON ---
        self.btn = QPushButton("INITIALIZE SYSTEM")
        self.btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn.setStyleSheet("""
            QPushButton { 
                background-color: #FF0055; 
                color: #FFFFFF; 
                font-size: 15px; 
                font-weight: bold; 
                letter-spacing: 1px;
                padding: 16px; 
                border-radius: 6px; 
                border: none;
            }
            QPushButton:hover { background-color: #E6004C; }
            QPushButton:pressed { background-color: #B3003B; }
        """)
        
        # Add Red Neon Glow to the Button
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor("#FF0055"))
        shadow.setOffset(0, 0)
        self.btn.setGraphicsEffect(shadow)
        
        self.btn.clicked.connect(self.trigger_boot_sequence)
        panel_layout.addWidget(self.btn)
        
        main_layout.addWidget(self.panel)
        
        # --- BOTTOM STATUS BAR ---
        self.status = QLabel("[ SYS.STATUS ] : AWAITING COMMAND")
        self.status.setStyleSheet("color: #475569; font-size: 11px; font-weight: bold;")
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.status)
        
        self.setLayout(main_layout)

    def trigger_boot_sequence(self):
        """Simulates a quick system check before launching for premium feel."""
        self.btn.setEnabled(False)
        self.btn.setText("CONNECTING NEURAL MESH...")
        self.status.setText("[ SYS.STATUS ] : ALLOCATING MEMORY...")
        self.status.setStyleSheet("color: #F59E0B; font-size: 11px; font-weight: bold;") # Amber/Warning color
        
        # Force UI update
        QApplication.processEvents()
        
        # Use QTimer to wait 600ms before actually launching, adding to the "heavy machinery" feel
        QTimer.singleShot(600, self.launch_system)

    def launch_system(self):
        idx = str(self.combo.currentIndex())
        self.status.setText(f"[ SYS.STATUS ] : PIPELINE ACTIVE ON PORT {idx}")
        self.status.setStyleSheet("color: #10B981; font-size: 11px; font-weight: bold;") # Green success color
        self.btn.setText("SYSTEM ONLINE")
        self.btn.setStyleSheet("QPushButton { background-color: #10B981; color: white; padding: 16px; border-radius: 6px; font-weight: bold; letter-spacing: 1px; }")
        
        # Change shadow to green
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor("#10B981"))
        shadow.setOffset(0, 0)
        self.btn.setGraphicsEffect(shadow)
        
        QApplication.processEvents()
        
        try:
            subprocess.Popen([self.venv_python, self.target_script, "--camera", idx])
            
            # Reset the button after 3 seconds so they can launch again if needed
            QTimer.singleShot(3000, self.reset_ui)
        except Exception as e:
            self.status.setText("[ SYS.ERROR ] : BOOT FAILURE. CHECK VENV.")
            self.status.setStyleSheet("color: #EF4444; font-size: 11px; font-weight: bold;")

    def reset_ui(self):
        self.btn.setEnabled(True)
        self.btn.setText("INITIALIZE SYSTEM")
        self.btn.setStyleSheet("""
            QPushButton { 
                background-color: #FF0055; 
                color: #FFFFFF; font-size: 15px; font-weight: bold; letter-spacing: 1px;
                padding: 16px; border-radius: 6px; border: none;
            }
            QPushButton:hover { background-color: #E6004C; }
        """)
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor("#FF0055"))
        shadow.setOffset(0, 0)
        self.btn.setGraphicsEffect(shadow)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DDSHLauncher()
    ex.show()
    sys.exit(app.exec())