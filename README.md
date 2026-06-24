<div align="center">

# 🦺 SafeStep
### Intelligent Fall Detection & Location-Aware Emergency Response System

**MediaPipe · ResNet18 · TTS/STT · Real-time Vision**

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-00897B?style=flat-square&logo=google&logoColor=white)](https://mediapipe.dev/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

<br>

> 📌 **2026학년도 오픈소스프로그래밍 프로젝트** | 팀명: 오르락 내리락

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Background & Motivation](#-background--motivation)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Theoretical Background](#-theoretical-background)
  - [Module 1: Fall Detection (MediaPipe Pose)](#module-1-fall-detection--mediapipe-pose)
  - [Module 2: Location Classification (ResNet18 CNN)](#module-2-location-classification--resnet18-cnn)
  - [Module 3: Voice Interaction (TTS/STT)](#module-3-voice-interaction--ttsstt)
- [System Flow](#-system-flow)
- [Dataset](#-dataset)
- [Requirements](#-requirements)
- [Installation & Quick Start](#-installation--quick-start)
- [Results](#-results)
- [Limitations](#-limitations)
- [Project Structure](#-project-structure)
- [Team](#-team)

---

## 📖 Overview

**SafeStep**은 고령자의 낙상 사고를 실시간으로 감지하고, 사고 발생 **장소(계단/평지)** 에 따라 차별화된 응급 대응을 자동으로 수행하는 지능형 시스템입니다.

기존 낙상 감지 기술의 한계:
- ❌ 단순히 '넘어짐' 자체만 판단 — 장소의 위험도를 고려하지 않음
- ❌ 계단 낙상(중증외상 위험)과 평지 낙상(경미한 경우)을 동일하게 처리
- ❌ 불필요한 행정력 낭비 및 정작 위험한 상황에서의 대응 지연

SafeStep의 해결책:
- ✅ **MediaPipe Pose** 기반 실시간 관절 추적으로 낙상 즉시 감지
- ✅ **ResNet18 CNN 전이 학습**으로 계단/평지 자동 분류
- ✅ 장소별 **차별화된 TTS/STT 응급 프로토콜** — 계단 낙상 시 즉시 119 신고

---

## 🔍 Background & Motivation

질병관리청 「제12차 국가 손상 종합 통계 2020」에 따르면, **60세 이상 입원 환자의 주요 손상기전은 추락·낙상(33.1%)** 으로 나타났습니다.

| 통계 | 수치 | 출처 |
|---|---|---|
| 65세 이상 고령자 안전사고 중 낙상 비율 | **47.4%** | 한국소비자원 CISS, 2017 |
| 65~84세 낙상 사망원인 중 계단 추락 비율 | **14.9%** | 통계청, 2024 |
| 고령자 안전사고 건수 (2016년) | **5,795건** | 한국소비자원 CISS, 2017 |

> 💡 **핵심 인사이트**: 평지 낙상은 경미한 경우가 많아 본인 확인 후 대응이 가능하지만, **계단 낙상은 복합골절·뇌손상 등 중증외상으로 이어질 가능성이 높아 즉각적인 응급 대응이 필수**입니다.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🦴 **실시간 낙상 감지** | MediaPipe Pose로 33개 관절 좌표를 실시간 추출하여 낙상 판별 |
| 🏗️ **장소 자동 분류** | ResNet18 CNN 전이 학습으로 계단(위험)/평지(저위험) 분류 |
| 🔊 **양방향 음성 인터랙션** | Google TTS/STT로 낙상 후 사용자 상태 확인 및 자동 응답 |
| 🚨 **차별화된 응급 프로토콜** | 계단 낙상 시 즉시 119 및 보호자 긴급 알림 자동 전송 |
| 📊 **실시간 모니터링 대시보드** | 낙상 발생 시간, 장소, 대응 현황 실시간 표시 |

---

## 🏗 System Architecture

SafeStep은 **3개의 독립 모듈**이 유기적으로 연동되는 파이프라인 구조입니다.
