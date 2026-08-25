"""Streamlit dashboard for live LangGraph lab demonstrations."""

from __future__ import annotations

import base64
import json
import os
from collections.abc import Callable
from html import escape
from pathlib import Path
from time import sleep
from typing import cast
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.metrics import (
    MetricsReport,
    ScenarioMetric,
    metric_from_state,
    summarize_metrics,
    write_metrics,
)
from langgraph_agent_lab.persistence import build_checkpointer
from langgraph_agent_lab.report import render_report, write_report
from langgraph_agent_lab.scenarios import load_scenarios
from langgraph_agent_lab.state import AgentState, Route, Scenario, initial_state

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
SAMPLE_SCENARIOS = ROOT / "data" / "sample" / "scenarios.jsonl"
UI_CHECKPOINTS = ROOT / "outputs" / "ui_checkpoints.db"
UI_METRICS = ROOT / "outputs" / "ui_metrics.json"
UI_REPORT = ROOT / "reports" / "ui_lab_report.md"
PIXEL_WORLD_IMAGE = ROOT / "src" / "langgraph_agent_lab" / "assets" / "orbit-pixel-world-v2.png"
PIXEL_WORLD_VIDEO = (
    ROOT / "src" / "langgraph_agent_lab" / "assets" / "orbit-pixel-world-loop-v2.webm"
)

NODE_LABELS = {
    "START": "Bắt đầu",
    "intake": "Tiếp nhận",
    "classify": "Hiểu ý định",
    "answer": "Trả lời",
    "tool": "Dùng công cụ",
    "evaluate": "Kiểm tra kết quả",
    "clarify": "Hỏi thêm",
    "wait_for_user": "Chờ bạn",
    "risky_action": "Chuẩn bị hành động",
    "approval": "Chờ duyệt",
    "retry": "Thử lại",
    "dead_letter": "Chuyển nhân viên",
    "finalize": "Hoàn tất",
    "END": "Kết thúc",
}

PRESETS: dict[str, dict[str, object]] = {
    "Câu hỏi đơn giản": {
        "query": "Mình quên mật khẩu rồi, giờ phải làm sao để đăng nhập lại?",
        "route": Route.SIMPLE,
        "max_attempts": 3,
    },
    "Tra cứu bằng công cụ": {
        "query": "Bạn xem giúp đơn hàng #12345 của mình đang giao tới đâu rồi nhé?",
        "route": Route.TOOL,
        "max_attempts": 3,
    },
    "Thiếu thông tin": {
        "query": "Đơn hàng của mình có vấn đề, bạn xử lý giúp mình được không?",
        "route": Route.MISSING_INFO,
        "max_attempts": 3,
    },
    "Theo dõi đơn nhiều lượt": {
        "query": "Đơn mình đặt vẫn chưa thấy giao tới. Bạn kiểm tra giúp mình nhé?",
        "route": Route.MISSING_INFO,
        "max_attempts": 3,
    },
    "Hoàn tiền cần HITL": {
        "query": (
            "Đơn #12345 bị giao nhầm hàng. Bạn hoàn lại 450.000 đồng vào thẻ "
            "và báo qua email giúp mình nhé."
        ),
        "route": Route.RISKY,
        "max_attempts": 3,
    },
    "Lỗi tạm thời và phục hồi": {
        "query": "Bạn kiểm tra giúp đơn hàng #12345 đang được giao tới đâu rồi nhé?",
        "route": Route.TOOL,
        "should_retry": True,
        "max_attempts": 3,
    },
    "Hết lượt thử": {
        "query": (
            "Từ sáng đến giờ mình mở mục hỗ trợ lần nào cũng thấy báo dịch vụ unavailable. "
            "Nếu vẫn không kết nối được thì chuyển giúp mình cho nhân viên nhé."
        ),
        "route": Route.ERROR,
        "max_attempts": 1,
    },
    "Tùy chỉnh": {"query": "", "route": Route.SIMPLE, "max_attempts": 3},
}

GRAPH_DOT = """
digraph LangGraph {
  rankdir=LR;
  graph [bgcolor="transparent", pad="0.3", nodesep="0.35", ranksep="0.6"];
  node [shape=box, style="rounded,filled", fillcolor="#0D1E32", color="#31536B",
        fontcolor="#DDF5E6", fontname="Courier New", fontsize=10, penwidth=1.4];
  edge [color="#31536B", fontcolor="#88A5B1", fontname="Courier New", fontsize=9];
  START [shape=circle, fillcolor="#102C2B", color="#69E3A7"];
  END [shape=doublecircle, fillcolor="#102C2B", color="#69E3A7"];
  classify [fillcolor="#102B35", color="#6CE5D1"];
  approval [fillcolor="#352B1D", color="#F8C763"];
  retry [fillcolor="#3A2027", color="#FF806F"];
  dead_letter [fillcolor="#3A2027", color="#FF806F"];
  START -> intake -> classify;
  classify -> answer [label="simple"];
  classify -> tool [label="tool"];
  classify -> clarify [label="missing_info"];
  classify -> risky_action [label="risky"];
  classify -> retry [label="error"];
  risky_action -> approval;
  approval -> tool [label="approved"];
  approval -> clarify [label="rejected"];
  tool -> evaluate;
  evaluate -> answer [label="success"];
  evaluate -> retry [label="needs_retry"];
  evaluate -> clarify [label="missing_info"];
  retry -> tool [label="within budget"];
  retry -> dead_letter [label="exhausted"];
  answer -> finalize;
  clarify -> wait_for_user;
  wait_for_user -> classify [label="user replied"];
  wait_for_user -> finalize [label="batch mode"];
  dead_letter -> finalize;
  finalize -> END;
}
"""

GRAPH_NODES = (
    "START",
    "intake",
    "classify",
    "answer",
    "tool",
    "evaluate",
    "clarify",
    "wait_for_user",
    "risky_action",
    "approval",
    "retry",
    "dead_letter",
    "finalize",
    "END",
)

GRAPH_EDGES = (
    ("START", "intake", ""),
    ("intake", "classify", ""),
    ("classify", "answer", "simple"),
    ("classify", "tool", "tool"),
    ("classify", "clarify", "missing_info"),
    ("classify", "risky_action", "risky"),
    ("classify", "retry", "error"),
    ("risky_action", "approval", ""),
    ("approval", "tool", "approved"),
    ("approval", "clarify", "rejected"),
    ("tool", "evaluate", ""),
    ("evaluate", "answer", "success"),
    ("evaluate", "retry", "needs_retry"),
    ("evaluate", "clarify", "missing_info"),
    ("retry", "tool", "within budget"),
    ("retry", "dead_letter", "exhausted"),
    ("answer", "finalize", ""),
    ("clarify", "wait_for_user", ""),
    ("wait_for_user", "classify", "user replied"),
    ("wait_for_user", "finalize", "batch mode"),
    ("dead_letter", "finalize", ""),
    ("finalize", "END", ""),
)

GRAPH_LAYOUT = {
    "START": (55, 280),
    "intake": (175, 280),
    "classify": (310, 280),
    "answer": (745, 70),
    "tool": (535, 155),
    "evaluate": (745, 155),
    "clarify": (535, 280),
    "wait_for_user": (745, 280),
    "risky_action": (440, 405),
    "approval": (620, 405),
    "retry": (440, 510),
    "dead_letter": (620, 510),
    "finalize": (925, 280),
    "END": (1050, 280),
}


@st.cache_data(show_spinner=False)
def _asset_data_uri(path: str, mime_type: str) -> str:
    """Embed a small project-owned media asset without requiring a static server."""
    payload = Path(path).read_bytes()
    return f"data:{mime_type};base64,{base64.b64encode(payload).decode('ascii')}"


def _render_video_welcome() -> None:
    """Render the chat entry point as a moving pixel-game world."""
    video_uri = _asset_data_uri(str(PIXEL_WORLD_VIDEO), "video/webm")
    poster_uri = _asset_data_uri(str(PIXEL_WORLD_IMAGE), "image/png")
    st.markdown(
        f"""
        <section class="world-video">
          <video autoplay muted loop playsinline poster="{poster_uri}">
            <source src="{video_uri}" type="video/webm">
          </video>
          <div class="world-shade"></div>
          <div class="world-copy">
            <span class="world-status"><i></i> TRẠM ORBIT ĐANG TRỰC</span>
            <h2>Mỗi yêu cầu là một hành trình.</h2>
            <p>Nhắn như đang nói chuyện với một nhân viên hỗ trợ. Orbit sẽ tự hỏi thêm,
            tra cứu hoặc xin phê duyệt khi cần.</p>
            <div class="world-hints">
              <span>① Bạn nhắn</span>
              <span>② Agent tự chọn đường</span>
              <span>③ Bạn nhận kết quả</span>
            </div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {--ink:#17211b; --muted:#66736a; --line:#dfe7df; --mint:#dff6df;
               --lime:#d9f99d; --cream:#f7f4ec; --forest:#214e34; --coral:#ff7657;}
        .stApp {background:#f7f7f2; color:var(--ink);}
        .block-container {padding-top:1rem; padding-bottom:3rem; max-width:1580px;}
        [data-testid="stSidebar"], [data-testid="collapsedControl"] {display:none !important;}
        [data-testid="stMetric"] {
            background:#fff; border:1px solid var(--line); border-radius:16px;
            padding:12px 16px; box-shadow:0 3px 12px rgba(31,55,40,.04);
        }
        .hero {
            padding:26px 30px; border-radius:24px; margin-bottom:16px;
            background:radial-gradient(circle at 80% 0%, #d9f99d 0, transparent 28%),
                       linear-gradient(120deg, #173e2a 0%, #295a3d 100%);
            color:white; box-shadow:0 14px 34px rgba(33,78,52,.18);
        }
        .hero h1 {margin:0 0 7px; font-size:2.15rem; color:white; letter-spacing:-.03em;}
        .hero p {margin:0; opacity:.82;}
        .eyebrow {font-size:.72rem; text-transform:uppercase; letter-spacing:.14em;
                  font-weight:800; color:#b9e8b8; margin-bottom:8px;}
        .pill {display:inline-block; padding:4px 10px; border-radius:999px;
               background:#eaf5e8; color:#214e34; font-weight:700; margin-right:6px;}
        .trace-title {font-weight:700; color:#344054; margin-top:8px;}
        .panel-title {font-size:.72rem; text-transform:uppercase; letter-spacing:.12em;
                      color:#758078; font-weight:800; margin:4px 0 10px;}
        .request-card {background:#fff; border:1px solid var(--line); border-radius:18px;
                       padding:14px 16px; margin:0 0 10px;}
        .request-card strong {color:var(--forest);}
        .journey-step {border-left:2px solid #cbd7cd; padding:2px 0 13px 14px;
                       color:#66736a; font-size:.86rem;}
        .journey-step.active {border-color:#55a76a; color:#173e2a; font-weight:700;}
        .journey-step.error {border-color:#ef6a55; color:#a33b2c;}
        .journey-dot {display:inline-block; width:8px; height:8px; border-radius:50%;
                      background:#55a76a; margin-right:7px;}
        div[data-testid="stGraphVizChart"] {background:#fff; border:1px solid var(--line);
                                             border-radius:20px; padding:12px;}
        .stButton > button {border-radius:999px; font-weight:700;}
        div[data-testid="stTabs"] button {font-weight:700;}
        .chat-shell {max-width:1050px; margin:0 auto;}
        .chat-head {display:flex; align-items:center; justify-content:space-between;
                    padding:10px 2px 18px; border-bottom:1px solid var(--line);}
        .agent-lockup {display:flex; align-items:center; gap:12px;}
        .agent-orb {width:44px; height:44px; border-radius:15px; display:grid; place-items:center;
                    color:#173e2a; font-size:1.2rem; font-weight:900;
                    background:radial-gradient(circle at 30% 25%,#fff 0 8%,transparent 9%),
                               linear-gradient(145deg,#f9e879,#9becb0 55%,#72cbd0);
                    box-shadow:0 8px 24px rgba(57,122,78,.22); transform:rotate(-3deg);}
        .agent-name {font-size:1.06rem; font-weight:850; letter-spacing:-.02em;}
        .agent-status {font-size:.78rem; color:#718077;}
        .online-dot {display:inline-block; width:7px; height:7px; border-radius:50%;
                     margin-right:5px; background:#35a854; box-shadow:0 0 0 4px #dff6df;}
        .mode-chip {padding:7px 11px; border:1px solid var(--line); border-radius:999px;
                    background:#fff; color:#536159; font-size:.75rem; font-weight:750;}
        .welcome-stage {position:relative; overflow:hidden; min-height:290px; margin:18px 0;
                        padding:46px 34px 30px; border-radius:32px; color:#183324;
                        background:radial-gradient(circle at 82% 18%,#fff6b8 0 6%,transparent 26%),
                                   radial-gradient(circle at 12% 100%,#a6e4de 0,transparent 37%),
                                   linear-gradient(135deg,#e2f5d5,#f7eedb 60%,#f8d7c5);}
        .welcome-stage:after {content:""; position:absolute; width:210px; height:210px;
                              border:42px solid rgba(255,255,255,.34);
                              border-radius:48% 52% 54% 46%;
                              right:-62px; bottom:-105px; transform:rotate(22deg);}
        .welcome-kicker {font-size:.73rem; text-transform:uppercase; letter-spacing:.14em;
                         font-weight:850; color:#407052;}
        .welcome-stage h2 {max-width:650px; margin:9px 0 8px; font-size:2.7rem;
                           line-height:1.02; letter-spacing:-.055em; color:#173e2a;}
        .welcome-stage p {max-width:620px; color:#50645a; font-size:1rem;}
        [data-testid="stChatMessage"] {background:transparent; padding:.5rem 0;}
        [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] {
            border-radius:22px; padding:14px 18px;}
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"])
          [data-testid="stChatMessageContent"] {background:#234c35; color:#fff;}
        .run-ribbon {display:flex; flex-wrap:wrap; gap:7px; align-items:center; margin:8px 0 5px;}
        .run-chip {display:inline-flex; gap:6px; align-items:center; border-radius:999px;
                   background:#edf4eb; color:#315c40; padding:5px 9px; font-size:.72rem;
                   font-weight:780; border:1px solid #d8e7d8;}
        .run-chip.warn {background:#fff0d5; color:#89561c; border-color:#f1d69f;}
        .run-chip.error {background:#ffe8e2; color:#983f32; border-color:#f2c5bb;}
        .activity-card {border:1px solid var(--line); background:rgba(255,255,255,.72);
                        border-radius:20px; padding:4px 14px 10px; margin:10px 0 4px;}
        .approval-card {border:1px solid #e8c77c; border-radius:20px; padding:16px 18px;
                        background:linear-gradient(135deg,#fff8df,#fff2e9); margin:10px 0;}
        .approval-card strong {color:#754a17;}
        [data-testid="stChatInput"] {border:1px solid #d8e2d8; border-radius:24px;
                                     background:rgba(255,255,255,.92);
                                     box-shadow:0 12px 35px rgba(41,76,53,.12);}
        @media (max-width:900px) {
          .welcome-stage h2 {font-size:2rem;} .welcome-stage {padding:30px 22px;}
        }
        .quest-header {display:flex; align-items:center; justify-content:space-between;
                       border:0; background:transparent; padding:2px 2px 12px;
                       box-shadow:none; margin:0; color:#294634; font:800 .72rem monospace;
                       letter-spacing:.08em;}
        .quest-live {color:#b83b2f; font-weight:900;
                     animation:pixel-blink 1s steps(2,end) infinite;}
        div[role="dialog"] {background:#e8f3e5 !important; border:4px solid #17251b !important;
                            border-radius:8px !important;
                            box-shadow:12px 12px 0 rgba(23,37,27,.34);}
        div[role="dialog"] div[data-testid="stGraphVizChart"] {border:3px solid #18251c;
                                                                border-radius:4px;
                                                                box-shadow:5px 5px 0 #b9cbb9;}
        /* V2 is a structural UI: moving world → focused mission map → optional inspector. */
        .world-video {position:relative; min-height:390px; margin:18px 0 22px; overflow:hidden;
                      border-radius:28px; background:#061228; isolation:isolate;
                      box-shadow:0 24px 70px rgba(5,18,39,.24);}
        .world-video video {position:absolute; inset:0; width:100%; height:100%; object-fit:cover;
                            image-rendering:auto; transform:scale(1.015);}
        .world-shade {position:absolute; inset:0;
                      background:linear-gradient(90deg,rgba(3,13,31,.92) 0%,rgba(4,18,36,.62) 48%,
                                                 rgba(4,13,28,.12) 78%),
                                 linear-gradient(0deg,rgba(2,10,25,.55),transparent 52%);}
        .world-copy {position:relative; z-index:2; width:min(610px,70%); padding:52px 46px;
                     color:#eef9ef;}
        .world-status {display:inline-flex; align-items:center; gap:8px; padding:7px 10px;
                       border:1px solid rgba(159,236,194,.42); border-radius:999px;
                       background:rgba(5,24,37,.58); backdrop-filter:blur(10px);
                       color:#bdf6d0; font:800 .68rem/1 monospace; letter-spacing:.1em;}
        .world-status i {width:7px; height:7px; border-radius:50%; background:#76f2aa;
                         box-shadow:0 0 0 5px rgba(118,242,170,.13),0 0 18px #76f2aa;
                         animation:world-signal 1.8s ease-in-out infinite;}
        .world-copy h2 {max-width:540px; margin:20px 0 12px; color:#fff;
                        font-size:clamp(2rem,4vw,3.35rem); line-height:.98; letter-spacing:-.06em;}
        .world-copy p {max-width:520px; margin:0; color:#c6d7d1; font-size:.96rem; line-height:1.6;}
        .world-hints {display:flex; flex-wrap:wrap; gap:8px; margin-top:25px;}
        .world-hints span {padding:7px 10px; border-radius:9px; color:#e3f1e8;
                           background:rgba(5,22,35,.7); border:1px solid rgba(255,255,255,.14);
                           font:700 .7rem monospace; backdrop-filter:blur(8px);}
        .mission-shell {position:relative; margin:14px 0 12px; padding:18px; overflow:hidden;
                        border:1px solid #294a61; border-radius:20px; color:#dff4e8;
                        background-color:#081528;
                        background-image:radial-gradient(circle at 12% 22%,#7af2bc 0 1px,
                                                          transparent 2px),
                                         radial-gradient(circle at 74% 18%,#f8c763 0 1px,
                                                          transparent 2px),
                                         linear-gradient(rgba(87,139,156,.07) 1px,transparent 1px),
                                         linear-gradient(90deg,rgba(87,139,156,.07) 1px,
                                                          transparent 1px);
                        background-size:83px 83px,127px 127px,18px 18px,18px 18px;
                        box-shadow:inset 0 0 45px rgba(30,98,100,.16),
                                   0 15px 40px rgba(5,18,39,.16);}
        .mission-shell:before {content:"ORBIT / NAV"; position:absolute; right:16px; bottom:9px;
                               color:rgba(122,196,180,.13); font:900 1.4rem monospace;
                               letter-spacing:.12em; pointer-events:none;}
        .mission-kicker {display:flex; justify-content:space-between; align-items:center;
                         position:relative; z-index:1; margin-bottom:13px; color:#7fa2ae;
                         font:800 .68rem monospace;
                         letter-spacing:.1em; text-transform:uppercase;}
        .mission-kicker b {color:#92efb5; text-shadow:0 0 12px rgba(105,227,167,.35);}
        .mission-track {display:flex; align-items:stretch; gap:24px; overflow-x:auto;
                        position:relative; z-index:1; padding:7px 6px 13px; scrollbar-width:thin;
                        scrollbar-color:#31536b transparent;}
        .mission-step {position:relative; flex:0 0 132px; min-height:70px;
                       padding:12px 11px 10px 39px;
                       border:1px solid #29465c; border-radius:4px;
                       background:rgba(9,27,44,.92); color:#7892a1;
                       box-shadow:3px 3px 0 rgba(2,9,20,.72);
                       transition:transform .25s ease,border-color .25s ease;}
        .mission-step:before {content:""; position:absolute; left:13px; top:26px;
                              width:9px; height:9px;
                              border:2px solid #43647a; background:#0b1c30;
                              box-shadow:0 0 0 4px rgba(62,104,125,.1); transform:rotate(45deg);}
        .mission-step:after {content:""; position:absolute; left:100%; top:31px; width:25px;
                             height:2px; background:#29465c; transform-origin:left;}
        .mission-step:last-child:after {display:none;}
        .mission-step.done {color:#b9e8cf; border-color:#3b7d70; background:rgba(13,48,49,.92);}
        .mission-step.done:before {border-color:#69e3a7; background:#69e3a7;
                                   box-shadow:0 0 0 4px rgba(105,227,167,.1),0 0 14px #69e3a7;}
        .mission-step.done:after {background:linear-gradient(90deg,#69e3a7,#4ba7a2,#69e3a7);
                                  background-size:200% 100%;
                                  animation:route-draw .48s ease both,
                                            route-scan 1.5s linear infinite;}
        .mission-step.router.done {border-color:#4ba7b4; color:#c4f6ef;}
        .mission-step.router.done:before {border-color:#6ce5d1; background:#6ce5d1;
                                          box-shadow:0 0 14px #6ce5d1;}
        .mission-step.human.done {border-color:#b78a40; color:#ffe8a5;}
        .mission-step.human.done:before {border-color:#f8c763; background:#f8c763;
                                         box-shadow:0 0 14px #f8c763;}
        .mission-step.danger.done {border-color:#9d4c50; color:#ffc1b4;
                                   background:rgba(52,27,36,.94);}
        .mission-step.danger.done:before {border-color:#ff806f; background:#ff806f;
                                          box-shadow:0 0 14px #ff806f;}
        .mission-step.danger.done:after {
            background:linear-gradient(90deg,#ff806f,#b34f55,#ff806f);
            background-size:200% 100%;}
        .mission-step.active {color:#fff3c4; border-color:#ffd56a; background:rgba(57,43,21,.94);
                              transform:translateY(-3px); animation:node-land .55s ease-out,
                              node-breathe 1.8s ease-in-out .55s infinite;}
        .mission-step.active:before {border-color:#ffd56a; background:#ffd56a;
                                     box-shadow:0 0 0 5px rgba(255,213,106,.12),0 0 20px #ffd56a;
                                     animation:beacon-lock .8s steps(2,end) infinite;}
        .mission-step.waiting {border-color:#ff9b77; background:rgba(61,31,29,.94);}
        .mission-step.waiting:before {border-color:#ff9b77; background:#ff9b77;
                                      box-shadow:0 0 16px #ff806f;}
        .mission-index {display:block; color:#6d8998; font:800 .58rem monospace;
                        letter-spacing:.07em;}
        .mission-name {display:block; margin-top:7px; font:800 .76rem/1.18 monospace;}
        .graph-world-scroll {width:100%; overflow-x:auto; padding-bottom:4px;
                             scrollbar-width:thin; scrollbar-color:#31536b transparent;}
        .graph-world {position:relative; width:100%; min-width:880px; aspect-ratio:1100/560;
                      overflow:hidden; border:1px solid #294a61; border-radius:18px;
                      background-color:#071326;
                      background-image:radial-gradient(circle at 12% 18%,#69e3a7 0 1px,
                                                        transparent 2px),
                                       radial-gradient(circle at 79% 31%,#f8c763 0 1px,
                                                        transparent 2px),
                                       linear-gradient(rgba(87,139,156,.06) 1px,transparent 1px),
                                       linear-gradient(90deg,rgba(87,139,156,.06) 1px,
                                                        transparent 1px);
                      background-size:91px 91px,139px 139px,20px 20px,20px 20px;
                      box-shadow:inset 0 0 70px rgba(30,98,100,.12),
                                 0 16px 42px rgba(5,18,39,.16);}
        .graph-world-head {position:absolute; z-index:4; left:17px; top:14px; right:17px;
                           display:flex; justify-content:space-between; color:#7899a7;
                           font:800 .62rem monospace; letter-spacing:.1em;
                           text-transform:uppercase;}
        .graph-world-head b {color:#92efb5;}
        .graph-world svg {position:absolute; inset:0; width:100%; height:100%; z-index:1;}
        .world-edge {fill:none; stroke:#26455b; stroke-width:2; opacity:.52;
                     marker-end:url(#orbit-arrow); transition:stroke .3s ease,opacity .3s ease;}
        .world-edge.visited {stroke:#69e3a7; stroke-width:3; opacity:1;
                             stroke-dasharray:8 6; marker-end:url(#orbit-arrow-live);
                             animation:edge-travel 1.15s linear infinite;}
        .world-edge.danger {stroke:#ff806f; marker-end:url(#orbit-arrow-danger);}
        .graph-node {position:absolute; z-index:2; width:112px; min-height:48px;
                     transform:translate(-50%,-50%); display:flex; flex-direction:column;
                     justify-content:center; padding:8px 10px; border:1px solid #29465c;
                     border-radius:5px; background:rgba(8,25,42,.96); color:#6f8998;
                     box-shadow:3px 3px 0 rgba(2,9,20,.72); text-align:center;
                     font-family:monospace; transition:all .28s ease;}
        .graph-node small {display:block; color:#496b7d; font-size:.5rem;
                           letter-spacing:.08em; text-transform:uppercase;}
        .graph-node strong {display:block; margin-top:3px; font-size:.67rem; line-height:1.15;}
        .graph-node.terminal {width:58px; min-height:58px; border-radius:50%; padding:4px;}
        .graph-node.visited {border-color:#54bd91; background:rgba(13,48,49,.97);
                             color:#d5f5e2; box-shadow:0 0 16px rgba(105,227,167,.25);}
        .graph-node.router.visited {border-color:#6ce5d1; color:#d2fbf5;
                                    box-shadow:0 0 18px rgba(108,229,209,.28);}
        .graph-node.human.visited {border-color:#f8c763; color:#ffe8a5;
                                   background:rgba(53,43,29,.97);}
        .graph-node.danger.visited {border-color:#ff806f; color:#ffc1b4;
                                    background:rgba(58,32,39,.97);}
        .graph-node.current {z-index:3; border-color:#ffd56a; color:#fff3c4;
                             background:rgba(59,48,28,.98);
                             transform:translate(-50%,-50%) scale(1.08);
                             animation:graph-lock .55s ease-out,
                                       node-radar 1.7s ease-in-out infinite;}
        .graph-node.current:after {content:""; position:absolute; inset:-7px;
                                   border:1px solid rgba(255,213,106,.58); border-radius:8px;
                                   animation:radar-ring 1.7s ease-out infinite;}
        .graph-node.terminal.current:after {border-radius:50%;}
        .graph-legend {position:absolute; z-index:4; right:16px; bottom:12px;
                       display:flex; gap:12px; color:#728d9a; font:700 .56rem monospace;}
        .graph-legend span:before {content:""; display:inline-block; width:7px; height:7px;
                                  margin-right:5px; border-radius:50%; background:#31536b;}
        .graph-legend .done:before {background:#69e3a7; box-shadow:0 0 8px #69e3a7;}
        .graph-legend .now:before {background:#ffd56a; box-shadow:0 0 8px #ffd56a;}
        .graph-legend .problem:before {background:#ff806f; box-shadow:0 0 8px #ff806f;}
        .focus-card {display:grid; grid-template-columns:auto 1fr; gap:13px; align-items:start;
                     margin:8px 0 12px; padding:15px 17px; border:1px solid #294a61;
                     border-radius:10px; background:#0b1d2e; color:#e9f4ec;
                     box-shadow:0 10px 28px rgba(5,18,39,.13);}
        .focus-pulse {width:12px; height:12px; margin-top:4px; border-radius:50%;
                      background:#ffe27c; box-shadow:0 0 0 7px rgba(255,226,124,.12);
                      animation:focus-ping 1.4s infinite;}
        .focus-node {color:#92efb5; font:850 .67rem monospace; letter-spacing:.1em;
                     text-transform:uppercase;}
        .focus-result {display:block; margin-top:5px; color:#f5faf6; font-size:.86rem;
                       line-height:1.45;}
        .trace-drawer {margin-top:8px; border-top:1px solid #dde5dd; color:#5d6b62;
                       font-size:.77rem;}
        .trace-drawer summary {padding:11px 2px 5px; cursor:pointer; font-weight:800;}
        .trace-line {display:grid; grid-template-columns:110px 1fr; gap:10px; padding:7px 2px;
                     border-bottom:1px dashed #dce4dc;}
        .trace-line b {font:800 .68rem monospace; color:#376449;}
        .turn-overview {margin:7px 0 4px; padding:12px 14px; border:1px solid #dde6de;
                        border-radius:15px; background:rgba(250,252,248,.9);}
        .turn-overview-head {display:flex; flex-wrap:wrap; gap:7px; align-items:center;
                             margin-bottom:9px;}
        .turn-overview-title {margin-right:auto; color:#244a32; font-size:.82rem; font-weight:850;}
        div[role="dialog"] {background:#f7f8f3 !important; border:1px solid #cfdacf !important;
                            border-radius:24px !important; box-shadow:0 30px 90px rgba(5,24,15,.3);}
        div[role="dialog"] div[data-testid="stGraphVizChart"] {
            border:1px solid #294a61; border-radius:12px; box-shadow:none;
            background:#081528;}
        div[data-testid="stGraphVizChart"] {border-color:#294a61 !important;
                                             background:#081528 !important;}
        .chat-shell div[data-testid="stGraphVizChart"] {border-color:#294a61;
                                                         background:#081528;}
        @keyframes world-signal {
            0%,100%{opacity:.55;transform:scale(.85)}
            50%{opacity:1;transform:scale(1)}}
        @keyframes route-draw {from{transform:scaleX(0)}to{transform:scaleX(1)}}
        @keyframes route-scan {from{background-position:200% 0}to{background-position:0 0}}
        @keyframes beacon-lock {0%,45%{opacity:1}46%,100%{opacity:.45}}
        @keyframes edge-travel {to{stroke-dashoffset:-28}}
        @keyframes graph-lock {from{opacity:.25;transform:translate(-50%,-50%) scale(.82)}
                               to{opacity:1;transform:translate(-50%,-50%) scale(1.08)}}
        @keyframes node-radar {0%,100%{box-shadow:0 0 0 0 rgba(255,213,106,0),
                                                  0 0 15px rgba(255,213,106,.16)}
                               50%{box-shadow:0 0 0 8px rgba(255,213,106,.08),
                                                   0 0 26px rgba(255,213,106,.35)}}
        @keyframes radar-ring {0%{opacity:.8;transform:scale(.9)}
                               100%{opacity:0;transform:scale(1.18)}}
        @keyframes node-land {0%{opacity:0;transform:translateY(10px) scale(.96)}
                              65%{transform:translateY(-5px) scale(1.02)}
                              100%{opacity:1;transform:translateY(-3px)}}
        @keyframes node-breathe {0%,100%{box-shadow:0 0 0 0 rgba(230,181,62,0)}
                                 50%{box-shadow:0 0 0 7px rgba(230,181,62,.12)}}
        @keyframes focus-ping {
            0%,100%{transform:scale(.85);opacity:.7}
            50%{transform:scale(1.15);opacity:1}}
        @keyframes pixel-float {0%,100%{transform:translateY(0)}50%{transform:translateY(8px)}}
        @keyframes pixel-blink {0%,45%{opacity:1}46%,100%{opacity:.25}}
        @media (max-width:760px) {
          .world-video {min-height:430px;} .world-copy {width:auto; padding:38px 24px;}
          .world-shade {background:linear-gradient(90deg,rgba(3,13,31,.92),rgba(4,18,36,.48));}
          .mission-step {flex-basis:112px;} .trace-line {grid-template-columns:80px 1fr;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def _cached_graph(checkpointer_kind: str) -> CompiledStateGraph:
    database_url = str(UI_CHECKPOINTS) if checkpointer_kind == "sqlite" else None
    return build_graph(build_checkpointer(checkpointer_kind, database_url))


def _configure_interrupts(
    enabled: bool, interactive_clarification: bool
) -> tuple[str | None, str | None]:
    previous = os.getenv("LANGGRAPH_INTERRUPT")
    previous_clarification = os.getenv("LANGGRAPH_CLARIFY_INTERRUPT")
    os.environ["LANGGRAPH_INTERRUPT"] = "true" if enabled else "false"
    os.environ["LANGGRAPH_CLARIFY_INTERRUPT"] = (
        "true" if interactive_clarification else "false"
    )
    return previous, previous_clarification


def _restore_interrupts(previous: tuple[str | None, str | None]) -> None:
    previous_hitl, previous_clarification = previous
    if previous_hitl is None:
        os.environ.pop("LANGGRAPH_INTERRUPT", None)
    else:
        os.environ["LANGGRAPH_INTERRUPT"] = previous_hitl
    if previous_clarification is None:
        os.environ.pop("LANGGRAPH_CLARIFY_INTERRUPT", None)
    else:
        os.environ["LANGGRAPH_CLARIFY_INTERRUPT"] = previous_clarification


def _invoke(
    graph: CompiledStateGraph,
    graph_input: AgentState | Command,
    config: RunnableConfig,
    *,
    real_hitl: bool,
    interactive_clarification: bool = False,
) -> AgentState:
    previous = _configure_interrupts(real_hitl, interactive_clarification)
    try:
        return cast(AgentState, graph.invoke(graph_input, config=config))
    finally:
        _restore_interrupts(previous)


def _stream_invoke(
    graph: CompiledStateGraph,
    graph_input: AgentState | Command,
    config: RunnableConfig,
    *,
    real_hitl: bool,
    interactive_clarification: bool = False,
    on_state: Callable[[AgentState], None],
) -> AgentState:
    """Stream state after every graph superstep for true node-by-node rendering."""
    previous = _configure_interrupts(real_hitl, interactive_clarification)
    latest = cast(AgentState, graph_input) if isinstance(graph_input, dict) else AgentState()
    try:
        for chunk in graph.stream(graph_input, config=config, stream_mode="values"):
            latest = cast(AgentState, chunk)
            on_state(latest)
        snapshot = graph.get_state(config)
        persisted = cast(AgentState, snapshot.values)
        if snapshot.interrupts:
            cast(dict[str, object], persisted)["__interrupt__"] = snapshot.interrupts
        return persisted or latest
    finally:
        _restore_interrupts(previous)


def _event_rows(state: AgentState) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, event in enumerate(state.get("events", []), start=1):
        metadata = event.get("metadata", {}) or {}
        rows.append(
            {
                "#": index,
                "node": event.get("node", "unknown"),
                "event": event.get("event_type", ""),
                "message": event.get("message", ""),
                "latency_ms": event.get("latency_ms", 0),
                "llm_calls": metadata.get("llm_calls", 0),
                "mode": metadata.get("evaluation_mode", ""),
            }
        )
    return rows


def _history_rows(graph: CompiledStateGraph, config: RunnableConfig) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, snapshot in enumerate(graph.get_state_history(config), start=1):
        values = cast(dict[str, object], snapshot.values)
        rows.append(
            {
                "checkpoint": index,
                "created_at": snapshot.created_at or "",
                "next": ", ".join(snapshot.next) or "END",
                "route": values.get("route", ""),
                "attempt": values.get("attempt", 0),
                "events": len(cast(list[object], values.get("events", []))),
            }
        )
    return rows


def _has_interrupt(state: AgentState) -> bool:
    return bool(cast(dict[str, object], state).get("__interrupt__"))


def _is_clarification_interrupt(state: AgentState) -> bool:
    """Identify a conversational pause without coupling the UI to interrupt internals."""
    return _has_interrupt(state) and bool(state.get("pending_question"))


def _interrupt_phase(state: AgentState) -> str:
    if _is_clarification_interrupt(state):
        return "clarification"
    if _has_interrupt(state):
        return "approval"
    return "complete"


def _execution_path(state: AgentState) -> list[str]:
    """Build the ordered node path represented by audit events and interrupts."""
    path = ["START"]
    path.extend(str(event.get("node", "unknown")) for event in state.get("events", []))
    if _has_interrupt(state):
        waiting_node = "wait_for_user" if _is_clarification_interrupt(state) else "approval"
        if not path or path[-1] != waiting_node:
            path.append(waiting_node)
    elif path and path[-1] == "finalize":
        path.append("END")
    return path


def _dynamic_graph_dot(state: AgentState, upto_step: int | None = None) -> str:
    """Render the graph with executed nodes and edges highlighted."""
    full_path = _execution_path(state)
    final_step = len(full_path) - 1 if upto_step is None else min(upto_step, len(full_path) - 1)
    visible_path = full_path[: final_step + 1]
    visited = set(visible_path)
    current = visible_path[-1]
    traversed_edges = set(zip(visible_path, visible_path[1:], strict=False))

    lines = [
        "digraph LangGraph {",
        '  rankdir=LR; graph [bgcolor="transparent", pad="0.35", nodesep="0.35", ranksep="0.65"];',
        '  node [shape=box, style="rounded,filled", fontname="Courier New", '
        'fontcolor="#DDF5E6", fontsize=10, margin="0.14,0.1"];',
        '  edge [fontname="Courier New", fontsize=9, arrowsize=0.75];',
    ]
    for node in GRAPH_NODES:
        shape = "doublecircle" if node == "END" else "circle" if node == "START" else "box"
        if node == current:
            fill, color, penwidth = "#3B301C", "#FFD56A", 3
        elif node in visited and node in {"retry", "dead_letter"}:
            fill, color, penwidth = "#3A2027", "#FF806F", 2
        elif node in visited and node == "approval":
            fill, color, penwidth = "#352B1D", "#F8C763", 2
        elif node in visited:
            fill, color, penwidth = "#102C2B", "#69E3A7", 2
        else:
            fill, color, penwidth = "#0D1E32", "#31536B", 1
        label = NODE_LABELS.get(node, node).replace(" ", "\\n")
        lines.append(
            f'  {node} [label="{label}", shape={shape}, fillcolor="{fill}", '
            f'color="{color}", penwidth={penwidth}];'
        )

    for source, target, label in GRAPH_EDGES:
        if (source, target) in traversed_edges:
            color = "#FF806F" if target in {"retry", "dead_letter"} else "#69E3A7"
            style = f'color="{color}", penwidth=3, fontcolor="{color}"'
        else:
            style = 'color="#31536B", penwidth=1, fontcolor="#88A5B1"'
        label_attr = f', label="{label}"' if label else ""
        lines.append(f"  {source} -> {target} [{style}{label_attr}];")
    lines.append("}")
    return "\n".join(lines)


def _path_caption(path: list[str], upto_step: int | None = None) -> str:
    final_step = len(path) - 1 if upto_step is None else min(upto_step, len(path) - 1)
    return " → ".join(path[: final_step + 1])


def _render_journey(state: AgentState) -> None:
    """Render a compact, human-readable journey for the selected request."""
    for index, event in enumerate(state.get("events", []), start=1):
        node = str(event.get("node", "unknown"))
        tone = "error" if node in {"retry", "dead_letter"} else "active"
        label = NODE_LABELS.get(node, node)
        message = str(event.get("message", ""))
        st.markdown(
            f'<div class="journey-step {tone}"><span class="journey-dot"></span>'
            f"<strong>{index:02d} · {label}</strong><br>{message}</div>",
            unsafe_allow_html=True,
        )
    if _has_interrupt(state):
        clarification = _is_clarification_interrupt(state)
        waiting_label = "Chờ bạn bổ sung" if clarification else "Chờ con người phê duyệt"
        waiting_message = (
            "Graph đã lưu checkpoint và sẽ tiếp tục sau khi bạn trả lời."
            if clarification
            else "Hành động chưa được thực hiện; graph đang dừng an toàn."
        )
        st.markdown(
            '<div class="journey-step active"><span class="journey-dot"></span>'
            f"<strong>{waiting_label}</strong><br>{waiting_message}</div>",
            unsafe_allow_html=True,
        )


def _remember_run(state: AgentState, expected_route: str) -> None:
    """Keep recent requests selectable so every run retains its own graph."""
    runs = cast(list[dict[str, object]], st.session_state.setdefault("recent_runs", []))
    thread_id = str(state.get("thread_id", "unknown"))
    entry: dict[str, object] = {
        "thread_id": thread_id,
        "state": state,
        "expected_route": expected_route,
    }
    runs[:] = [item for item in runs if item.get("thread_id") != thread_id]
    runs.insert(0, entry)
    del runs[8:]


def _select_visible_run(default_state: AgentState, default_route: str) -> tuple[AgentState, str]:
    """Let presenters switch between recent request-specific visualizations."""
    runs = cast(list[dict[str, object]], st.session_state.get("recent_runs", []))
    if len(runs) < 2:
        return default_state, default_route
    labels = [
        f"{str(cast(AgentState, item['state']).get('scenario_id', 'request'))} · "
        f"{str(cast(AgentState, item['state']).get('route', 'pending')) or 'pending'}"
        for item in runs
    ]
    selected = st.selectbox("Recent requests", range(len(runs)), format_func=labels.__getitem__)
    item = runs[selected]
    return cast(AgentState, item["state"]), str(item["expected_route"])


def _render_result(state: AgentState, expected_route: str) -> None:
    metric = metric_from_state(
        cast(dict[str, object], state),
        expected_route=expected_route,
        approval_required=expected_route == Route.RISKY.value,
    )
    events = state.get("events", [])
    total_latency = sum(int(event.get("latency_ms", 0) or 0) for event in events)

    st.markdown('<div class="panel-title">Request workspace</div>', unsafe_allow_html=True)
    scenario_label = escape(str(state.get("scenario_id", "Live request")))
    query_label = escape(str(state.get("query", "")))
    st.markdown(
        f'<div class="request-card"><strong>{scenario_label}</strong>'
        f"<br><span>{query_label}</span></div>",
        unsafe_allow_html=True,
    )

    journey_col, graph_col, inspector_col = st.columns([0.85, 2.25, 1.05], gap="medium")
    with journey_col:
        st.markdown('<div class="panel-title">Journey</div>', unsafe_allow_html=True)
        _render_journey(state)
    with graph_col:
        st.markdown('<div class="panel-title">Live graph</div>', unsafe_allow_html=True)
        st.graphviz_chart(_dynamic_graph_dot(state), width="stretch")
        st.caption(f"Executed path · {_path_caption(_execution_path(state))}")
    with inspector_col:
        st.markdown('<div class="panel-title">Run inspector</div>', unsafe_allow_html=True)
        st.metric("Status", "Waiting for human" if _has_interrupt(state) else "Completed")
        st.metric("Route", state.get("route", "pending") or "pending")
        st.metric("Attempts", state.get("attempt", 0))
        st.metric("Thread", str(state.get("thread_id", "unknown"))[-12:])

    cols = st.columns(6)
    cols[0].metric("Actual route", state.get("route", "pending") or "pending")
    cols[1].metric("Expected", expected_route)
    cols[2].metric("Nodes", len(events))
    cols[3].metric("Retries", metric.retry_count)
    cols[4].metric("LLM calls", metric.llm_calls)
    cols[5].metric("LLM latency", f"{total_latency:,} ms")

    if _has_interrupt(state):
        st.warning("Workflow is paused at the approval node and waiting for a human decision.")
    elif metric.success:
        st.success("Workflow completed with the expected behavior.")
    else:
        st.error("Workflow completed, but its behavior differs from the expected route/outcome.")

    answer = state.get("final_answer") or state.get("pending_question")
    if answer:
        st.subheader("Final response")
        st.info(answer)

    if state.get("proposed_action"):
        st.subheader("Proposed risky action")
        st.warning(state["proposed_action"])

    with st.expander("Open node-by-node execution trace", expanded=False):
        st.dataframe(_event_rows(state), width="stretch", hide_index=True)

    detail_tabs = st.tabs(["Tool results", "Errors", "State JSON", "Messages"])
    with detail_tabs[0]:
        tool_results = state.get("tool_results", [])
        st.code("\n\n".join(tool_results) if tool_results else "No tool result", language="text")
    with detail_tabs[1]:
        errors = state.get("errors", [])
        st.code("\n".join(errors) if errors else "No errors", language="text")
    with detail_tabs[2]:
        st.json(cast(dict[str, object], state), expanded=False)
    with detail_tabs[3]:
        st.code("\n".join(state.get("messages", [])) or "No messages", language="text")


def _render_pending_approval() -> None:
    if "pending_graph" not in st.session_state:
        return
    st.subheader("Human approval console")
    st.caption("This resumes the same checkpointed thread; it does not restart the graph.")
    comment = st.text_input("Reviewer comment", value="Reviewed during live demo")
    approve_col, reject_col, _ = st.columns([1, 1, 3])
    approved = approve_col.button("Approve", type="primary", width="stretch")
    rejected = reject_col.button("Reject", width="stretch")
    if not (approved or rejected):
        return

    graph = cast(CompiledStateGraph, st.session_state["pending_graph"])
    config = cast(RunnableConfig, st.session_state["pending_config"])
    decision = {
        "approved": approved,
        "reviewer": "streamlit-reviewer",
        "comment": comment,
    }
    with st.spinner("Resuming checkpointed workflow..."):
        result = _invoke(graph, Command(resume=decision), config, real_hitl=True)
    st.session_state["last_result"] = result
    _remember_run(result, str(st.session_state.get("last_expected_route", Route.RISKY.value)))
    st.session_state.pop("pending_graph", None)
    st.session_state.pop("pending_config", None)
    st.rerun()


def _run_single_demo(checkpointer_kind: str) -> None:
    left, right = st.columns([1.5, 1])
    with left:
        preset_name = st.selectbox("Demo preset", list(PRESETS))
        preset = PRESETS[preset_name]
        query = st.text_area(
            "Support ticket",
            value=str(preset["query"]),
            height=110,
            key=f"query-{preset_name}",
        )
    with right:
        routes = [route.value for route in Route if route not in {Route.DEAD_LETTER, Route.DONE}]
        preset_route = cast(Route, preset["route"]).value
        expected_route = st.selectbox(
            "Expected route",
            routes,
            index=routes.index(preset_route),
            key=f"route-{preset_name}",
        )
        preset_max_attempts = preset["max_attempts"]
        max_attempts = st.number_input(
            "Max attempts",
            min_value=1,
            max_value=5,
            value=preset_max_attempts if isinstance(preset_max_attempts, int) else 3,
        )
        real_hitl = st.toggle(
            "Real HITL interrupt",
            value=preset_route == Route.RISKY.value,
            help="Risky routes pause at approval and resume from the same checkpoint.",
        )

    run_clicked = st.button("Run workflow", type="primary", width="stretch")
    if run_clicked:
        if not query.strip():
            st.error("Enter a support ticket before running the graph.")
        else:
            scenario = Scenario(
                id=f"ui-{uuid4().hex[:8]}",
                query=query,
                expected_route=Route(expected_route),
                requires_approval=expected_route == Route.RISKY.value,
                max_attempts=int(max_attempts),
            )
            state = initial_state(scenario)
            graph = _cached_graph(checkpointer_kind)
            config: RunnableConfig = {"configurable": {"thread_id": state["thread_id"]}}
            with st.spinner("Invoking LangGraph and the configured LLM..."):
                result = _invoke(graph, state, config, real_hitl=real_hitl)
            st.session_state["last_result"] = result
            st.session_state["last_expected_route"] = expected_route
            st.session_state["last_config"] = config
            st.session_state["last_graph"] = graph
            if _has_interrupt(result):
                st.session_state["pending_graph"] = graph
                st.session_state["pending_config"] = config
            else:
                st.session_state.pop("pending_graph", None)
                st.session_state.pop("pending_config", None)
            _remember_run(result, expected_route)

    _render_pending_approval()
    if "last_result" in st.session_state:
        visible_state, visible_route = _select_visible_run(
            cast(AgentState, st.session_state["last_result"]),
            str(st.session_state.get("last_expected_route", expected_route)),
        )
        _render_result(visible_state, visible_route)


def _ensure_conversations() -> None:
    """Initialize the multi-chat workspace once per browser session."""
    if "conversations" in st.session_state:
        return
    chat_id = f"chat-{uuid4().hex[:8]}"
    st.session_state["conversations"] = {chat_id: {"title": "Cuộc trò chuyện mới", "turns": []}}
    st.session_state["active_chat_id"] = chat_id


def _active_conversation() -> tuple[str, dict[str, object]]:
    _ensure_conversations()
    chats = cast(dict[str, dict[str, object]], st.session_state["conversations"])
    active_id = str(st.session_state["active_chat_id"])
    return active_id, chats[active_id]


def _new_conversation() -> None:
    chats = cast(dict[str, dict[str, object]], st.session_state["conversations"])
    chat_id = f"chat-{uuid4().hex[:8]}"
    chats[chat_id] = {"title": "Cuộc trò chuyện mới", "turns": []}
    st.session_state["active_chat_id"] = chat_id


def _workspace_defaults() -> str:
    """Initialize the fixed live-demo configuration without rendering a sidebar."""
    _ensure_conversations()
    st.session_state["chat_hitl"] = True
    st.session_state["show_live_graph"] = True
    st.session_state["chat_max_attempts"] = 3
    return "sqlite"


def _conversation_toolbar() -> None:
    """Keep multi-chat navigation available in the content area."""
    chats = cast(dict[str, dict[str, object]], st.session_state["conversations"])
    active_id = str(st.session_state["active_chat_id"])
    chat_ids = list(reversed(chats))
    active_turns = cast(list[dict[str, object]], chats[active_id]["turns"])
    chat_labels = {
        chat_id: str(chats[chat_id].get("title", "Cuộc trò chuyện mới")) for chat_id in chat_ids
    }

    def format_chat_id(chat_id: str) -> str:
        return chat_labels[chat_id]

    selector_col, new_col = st.columns([5, 1.25], gap="small", vertical_alignment="bottom")
    selected = selector_col.selectbox(
        "Cuộc hội thoại",
        chat_ids,
        index=chat_ids.index(active_id),
        format_func=format_chat_id,
        label_visibility="collapsed",
        key=f"conversation-picker-{active_id}-{len(active_turns)}",
    )
    if selected != active_id:
        st.session_state["active_chat_id"] = selected
        st.rerun()
    if new_col.button("＋  NHIỆM VỤ MỚI", type="primary", width="stretch"):
        _new_conversation()
        st.rerun()


def _submit_chat_request(
    query: str,
    checkpointer_kind: str,
    *,
    preset_name: str | None = None,
    on_state: Callable[[AgentState], None] | None = None,
) -> None:
    """Run one message and attach its complete workflow evidence to the chat."""
    chat_id, chat = _active_conversation()
    turns = cast(list[dict[str, object]], chat["turns"])
    preset = PRESETS.get(preset_name or "Tùy chỉnh", PRESETS["Tùy chỉnh"])
    expected = cast(Route, preset["route"])
    preset_attempts = cast(int, preset.get("max_attempts", st.session_state["chat_max_attempts"]))
    max_attempts = (
        preset_attempts if preset_name else cast(int, st.session_state["chat_max_attempts"])
    )
    scenario = Scenario(
        id=f"{chat_id}-turn-{len(turns) + 1}",
        query=query,
        expected_route=expected,
        requires_approval=expected == Route.RISKY,
        should_retry=bool(preset.get("should_retry", False)),
        max_attempts=max_attempts,
    )
    state = initial_state(scenario)
    graph = _cached_graph(checkpointer_kind)
    config: RunnableConfig = {"configurable": {"thread_id": state["thread_id"]}}
    if on_state:
        result = _stream_invoke(
            graph,
            state,
            config,
            real_hitl=bool(st.session_state["chat_hitl"]),
            interactive_clarification=True,
            on_state=on_state,
        )
    else:
        result = _invoke(
            graph,
            state,
            config,
            real_hitl=bool(st.session_state["chat_hitl"]),
            interactive_clarification=True,
        )
    turn: dict[str, object] = {
        "query": query,
        "state": result,
        "expected_route": expected.value,
        "graph": graph,
        "config": config,
    }
    turns.append(turn)
    if len(turns) == 1:
        chat["title"] = query.strip()[:31] + ("…" if len(query.strip()) > 31 else "")

    st.session_state["last_result"] = result
    st.session_state["last_expected_route"] = expected.value
    st.session_state["last_config"] = config
    st.session_state["last_graph"] = graph
    _remember_run(result, expected.value)


def _resume_chat_request(
    chat_id: str,
    turn_index: int,
    approved: bool,
    comment: str,
    on_state: Callable[[AgentState], None] | None = None,
) -> AgentState:
    """Resume the exact interrupted turn and update it in place."""
    chats = cast(dict[str, dict[str, object]], st.session_state["conversations"])
    chat = chats[chat_id]
    turns = cast(list[dict[str, object]], chat["turns"])
    turn = turns[turn_index]
    graph = cast(CompiledStateGraph, turn["graph"])
    config = cast(RunnableConfig, turn["config"])
    decision = {
        "approved": approved,
        "reviewer": "live-demo-reviewer",
        "comment": comment,
    }
    if on_state:
        result = _stream_invoke(
            graph,
            Command(resume=decision),
            config,
            real_hitl=True,
            interactive_clarification=True,
            on_state=on_state,
        )
    else:
        with st.spinner("Đang tiếp tục từ checkpoint đã lưu..."):
            result = _invoke(
                graph,
                Command(resume=decision),
                config,
                real_hitl=True,
                interactive_clarification=True,
            )
    turn["state"] = result
    st.session_state["last_result"] = result
    _remember_run(result, str(turn["expected_route"]))
    return result


def _resume_chat_clarification(
    chat_id: str,
    turn_index: int,
    answer: str,
    on_state: Callable[[AgentState], None] | None = None,
) -> AgentState:
    """Feed a user's follow-up into the paused graph instead of creating a new run."""
    chats = cast(dict[str, dict[str, object]], st.session_state["conversations"])
    turns = cast(list[dict[str, object]], chats[chat_id]["turns"])
    turn = turns[turn_index]
    paused_state = cast(AgentState, turn["state"])
    question = str(paused_state.get("pending_question") or "Bạn có thể nói rõ hơn không?")
    graph = cast(CompiledStateGraph, turn["graph"])
    config = cast(RunnableConfig, turn["config"])
    graph_input: Command = Command(resume={"answer": answer})
    if on_state:
        result = _stream_invoke(
            graph,
            graph_input,
            config,
            real_hitl=bool(st.session_state["chat_hitl"]),
            interactive_clarification=True,
            on_state=on_state,
        )
    else:
        with st.spinner("Đang tiếp tục đúng workflow đã tạm dừng..."):
            result = _invoke(
                graph,
                graph_input,
                config,
                real_hitl=bool(st.session_state["chat_hitl"]),
                interactive_clarification=True,
            )
    turn["state"] = result
    turn.setdefault("clarifications", [])
    cast(list[dict[str, str]], turn["clarifications"]).append(
        {"question": question, "answer": answer}
    )
    st.session_state["last_result"] = result
    _remember_run(result, str(turn["expected_route"]))
    return result


def _run_tone(state: AgentState) -> str:
    if _has_interrupt(state):
        return "warn"
    if any(node in {"retry", "dead_letter"} for node in _execution_path(state)):
        return "error"
    return ""


def _render_turn_activity(state: AgentState, turn_number: int, *, expanded: bool) -> None:
    path = _execution_path(state)
    tone = _run_tone(state)
    route = str(state.get("route", "pending") or "pending")
    if _is_clarification_interrupt(state):
        status = "Đang chờ bạn trả lời"
    elif _has_interrupt(state):
        status = "Đang chờ phê duyệt"
    else:
        status = "Đã hoàn tất"
    st.markdown(
        '<div class="turn-overview"><div class="turn-overview-head">'
        f'<span class="turn-overview-title">Lượt {turn_number} · {escape(status)}</span>'
        f'<span class="run-chip {tone}">Route · {escape(route)}</span>'
        f'<span class="run-chip">{max(len(path) - 2, 0)} bước xử lý</span>'
        f'<span class="run-chip">Retry {state.get("attempt", 0)}</span>'
        "</div></div>",
        unsafe_allow_html=True,
    )
    if st.session_state.get("show_live_graph", True):
        st.markdown(_full_graph_map_html(state), unsafe_allow_html=True)
    with st.expander("Xem inspector kỹ thuật", expanded=False):
        summary_col, graph_col = st.columns([0.8, 2.2], gap="medium")
        with summary_col:
            _render_journey(state)
        with graph_col:
            st.graphviz_chart(_dynamic_graph_dot(state), width="stretch")
        st.dataframe(_event_rows(state), width="stretch", hide_index=True)
        if state.get("tool_results") or state.get("errors"):
            detail_col, error_col = st.columns(2)
            detail_col.code("\n\n".join(state.get("tool_results", [])) or "Không có kết quả")
            error_col.code("\n".join(state.get("errors", [])) or "Không có lỗi")


def _render_inline_approval(
    state: AgentState, turn_key: str, chat_id: str, turn_index: int
) -> None:
    # Clarification pauses are answered in chat and must never render approval controls.
    if not _has_interrupt(state) or _is_clarification_interrupt(state):
        return
    st.markdown(
        '<div class="approval-card"><strong>Checkpoint của con người</strong><br>'
        "Orbit đã chuẩn bị hành động có tác động thật và dừng trước khi thực hiện. "
        "Hãy kiểm tra rồi tiếp tục đúng workflow đã lưu.</div>",
        unsafe_allow_html=True,
    )
    if state.get("proposed_action"):
        st.warning(state["proposed_action"])
    comment = st.text_input(
        "Ghi chú của người duyệt",
        value="Đã xác minh danh tính khách hàng và chính sách trong bản demo",
        key=f"approval-note-{turn_key}",
    )
    approve_col, reject_col, _ = st.columns([1, 1, 2])
    if approve_col.button("Phê duyệt & tiếp tục", type="primary", key=f"approve-{turn_key}"):
        _resume_chat_request(chat_id, turn_index, True, comment)
        st.rerun()
    if reject_col.button("Từ chối", key=f"reject-{turn_key}"):
        _resume_chat_request(chat_id, turn_index, False, comment)
        st.rerun()


def _queue_chat_request(query: str, preset_name: str | None = None) -> None:
    """Queue a message so its workflow can run inside the live modal."""
    st.session_state["queued_chat_request"] = {
        "query": query,
        "preset_name": preset_name,
    }


def _node_result(state: AgentState, node: str, fallback: str) -> str:
    """Extract the most demo-worthy output produced by one node."""
    if node == "classify":
        return f"Đã chọn route: {state.get('route', 'đang chờ')}"
    if node == "tool" and state.get("tool_results"):
        return state["tool_results"][-1]
    if node == "evaluate":
        return f"Kết quả đánh giá: {state.get('evaluation_result', 'đang chờ')}"
    if node == "approval" and _has_interrupt(state):
        return "Hành động đã được chuẩn bị nhưng chưa thực hiện; đang chờ con người quyết định"
    if node in {"risky_action", "approval"} and state.get("proposed_action"):
        return str(state["proposed_action"])
    if node == "wait_for_user":
        if _is_clarification_interrupt(state):
            return str(state.get("pending_question") or "Agent đang chờ bạn bổ sung thông tin")
        return "Đã nhận câu trả lời và gửi lại yêu cầu vào bộ định tuyến"
    if node in {"answer", "clarify", "finalize"}:
        answer = state.get("final_answer") or state.get("pending_question")
        if answer:
            return str(answer)
    if node in {"retry", "dead_letter"} and state.get("errors"):
        return state["errors"][-1]
    return fallback


def _mission_map_html(state: AgentState) -> str:
    """Render the actual execution path as a calm, animated mission map."""
    path = _execution_path(state)
    current = path[-1]
    steps: list[str] = []
    for index, node in enumerate(path, start=1):
        classes = ["mission-step"]
        if node in {"retry", "dead_letter"}:
            classes.append("danger")
        elif node == "approval":
            classes.append("human")
        elif node == "classify":
            classes.append("router")
        if node == current:
            classes.append("active")
            if node in {"approval", "wait_for_user"}:
                classes.append("waiting")
        else:
            classes.append("done")
        steps.append(
            f'<div class="{" ".join(classes)}">'
            f'<span class="mission-index">BƯỚC {index:02d}</span>'
            f'<span class="mission-name">{escape(NODE_LABELS.get(node, node))}</span></div>'
        )
    route = escape(str(state.get("route", "đang xác định") or "đang xác định"))
    return (
        '<div class="mission-shell"><div class="mission-kicker">'
        f'<span>Hành trình hiện tại</span><b>ROUTE · {route}</b></div>'
        f'<div class="mission-track">{"".join(steps)}</div></div>'
    )


def _graph_edge_path(source: str, target: str) -> str:
    """Route one visual edge while keeping node boxes readable."""
    source_x, source_y = GRAPH_LAYOUT[source]
    target_x, target_y = GRAPH_LAYOUT[target]
    source_half = 32 if source in {"START", "END"} else 57
    target_half = 32 if target in {"START", "END"} else 57
    if target_x > source_x:
        start_x = source_x + source_half
        end_x = target_x - target_half
        control_1 = start_x + (end_x - start_x) * 0.42
        control_2 = start_x + (end_x - start_x) * 0.72
        return (
            f"M {start_x} {source_y} C {control_1:.1f} {source_y}, "
            f"{control_2:.1f} {target_y}, {end_x} {target_y}"
        )
    loop_y = max(source_y, target_y) + 58
    return (
        f"M {source_x} {source_y + 28} C {source_x} {loop_y}, "
        f"{target_x} {loop_y}, {target_x} {target_y + 28}"
    )


def _full_graph_map_html(state: AgentState) -> str:
    """Keep the whole topology visible and illuminate the path taken by the agent."""
    path = _execution_path(state)
    visited = set(path)
    current = path[-1]
    traversed_edges = set(zip(path, path[1:], strict=False))
    edges: list[str] = []
    for source, target, label in GRAPH_EDGES:
        classes = ["world-edge"]
        if (source, target) in traversed_edges:
            classes.append("visited")
            if target in {"retry", "dead_letter"}:
                classes.append("danger")
        title = escape(label or f"{source} → {target}")
        edges.append(
            f'<path class="{" ".join(classes)}" d="{_graph_edge_path(source, target)}">'
            f"<title>{title}</title></path>"
        )

    nodes: list[str] = []
    for node in GRAPH_NODES:
        x, y = GRAPH_LAYOUT[node]
        classes = ["graph-node"]
        if node in {"START", "END"}:
            classes.append("terminal")
        if node in {"retry", "dead_letter"}:
            classes.append("danger")
        elif node == "approval":
            classes.append("human")
        elif node == "classify":
            classes.append("router")
        if node in visited:
            classes.append("visited")
        if node == current:
            classes.append("current")
        nodes.append(
            f'<div class="{" ".join(classes)}" style="left:{x / 11:.3f}%;top:{y / 5.6:.3f}%">'
            f'<small>{escape(node)}</small><strong>{escape(NODE_LABELS.get(node, node))}</strong>'
            "</div>"
        )

    route = escape(str(state.get("route", "đang xác định") or "đang xác định"))
    return (
        '<div class="graph-world-scroll"><div class="graph-world">'
        '<div class="graph-world-head"><span>Toàn bộ graph</span>'
        f"<b>ROUTE · {route}</b></div>"
        '<svg viewBox="0 0 1100 560" aria-label="LangGraph topology">'
        '<defs><marker id="orbit-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" '
        'orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#31536b"/></marker>'
        '<marker id="orbit-arrow-live" markerWidth="7" markerHeight="7" refX="6" refY="3.5" '
        'orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#69e3a7"/></marker>'
        '<marker id="orbit-arrow-danger" markerWidth="7" markerHeight="7" refX="6" '
        'refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#ff806f"/></marker>'
        f'</defs>{"".join(edges)}</svg>{"".join(nodes)}'
        '<div class="graph-legend"><span class="done">Đã chạy</span>'
        '<span class="now">Hiện tại</span><span class="problem">Retry/lỗi</span></div>'
        "</div></div>"
    )


def _live_console_html(state: AgentState, notes: list[tuple[str, str]]) -> str:
    """Show one current decision; preserve prior node output in a collapsed drawer."""
    current = _execution_path(state)[-1]
    fallback = "Agent đang xử lý bước này"
    if state.get("events"):
        fallback = str(state["events"][-1].get("message", fallback))
    result = _node_result(state, current, fallback)
    history = []
    for node, detail in notes:
        history.append(
            '<div class="trace-line">'
            f'<b>{escape(NODE_LABELS.get(node, node).upper())}</b>'
            f'<span>{escape(detail[:500])}</span></div>'
        )
    return (
        '<div class="focus-card"><span class="focus-pulse"></span><div>'
        f'<span class="focus-node">ĐANG Ở · {escape(NODE_LABELS.get(current, current))}</span>'
        f'<span class="focus-result">{escape(result[:520])}</span></div></div>'
        '<details class="trace-drawer"><summary>Xem toàn bộ dữ liệu từng node</summary>'
        f'{"".join(history) or "<p>Chưa có dữ liệu node.</p>"}</details>'
    )


@st.dialog(
    "Orbit đang xử lý yêu cầu",
    width="large",
    dismissible=False,
    icon="🕹️",
)
def _live_workflow_dialog(checkpointer_kind: str) -> None:
    """Run the modal as a safe queued -> approval -> complete state machine."""
    if "queued_chat_request" not in st.session_state:
        st.info("Nhiệm vụ đã được đóng. Đang trở về cuộc hội thoại...")
        if st.button("Về cuộc hội thoại", type="primary", width="stretch"):
            st.rerun(scope="app")
        return

    queued = cast(dict[str, object], st.session_state["queued_chat_request"])
    query = str(queued["query"])
    preset_name = cast(str | None, queued.get("preset_name"))
    phase = str(queued.get("phase", "queued"))
    st.markdown(
        '<div class="quest-header"><strong>HÀNH TRÌNH CỦA YÊU CẦU</strong>'
        '<span class="quest-live">● ĐANG CHẠY</span></div>',
        unsafe_allow_html=True,
    )
    st.caption(query)
    status_slot = st.empty()
    graph_slot = st.empty()
    notes_slot = st.empty()
    notes = cast(list[tuple[str, str]], queued.setdefault("notes", []))
    seen_events = len(notes)

    def render_state(state: AgentState) -> None:
        nonlocal seen_events
        events = state.get("events", [])
        for event in events[seen_events:]:
            node = str(event.get("node", "unknown"))
            fallback = str(event.get("message", "Node đã hoàn tất"))
            notes.append((node, _node_result(state, node, fallback)))
        seen_events = len(events)
        current = _execution_path(state)[-1]
        current_label = NODE_LABELS.get(current, current)
        status_slot.caption(f"Bước hiện tại: {current_label} · {len(events)} sự kiện đã ghi nhận")
        graph_slot.markdown(_full_graph_map_html(state), unsafe_allow_html=True)
        notes_slot.markdown(_live_console_html(state, notes), unsafe_allow_html=True)
        if events:
            sleep(0.4)

    if phase == "queued":
        _submit_chat_request(
            query,
            checkpointer_kind,
            preset_name=preset_name,
            on_state=render_state,
        )
        chat_id, chat = _active_conversation()
        turns = cast(list[dict[str, object]], chat["turns"])
        queued["chat_id"] = chat_id
        queued["turn_index"] = len(turns) - 1
        latest_state = cast(AgentState, turns[-1]["state"])
        phase = _interrupt_phase(latest_state)
        queued["phase"] = phase
    else:
        chats = cast(dict[str, dict[str, object]], st.session_state["conversations"])
        chat = chats[str(queued["chat_id"])]
        turns = cast(list[dict[str, object]], chat["turns"])
        latest_state = cast(AgentState, turns[cast(int, queued["turn_index"])]["state"])
        render_state(latest_state)
        if phase == "resume_clarification":
            latest_state = _resume_chat_clarification(
                str(queued["chat_id"]),
                cast(int, queued["turn_index"]),
                str(queued["clarification_answer"]),
                on_state=render_state,
            )
            phase = _interrupt_phase(latest_state)
            queued["phase"] = phase

    if phase == "clarification":
        status_slot.info("ĐANG CHỜ BẠN // WORKFLOW ĐÃ LƯU CHECKPOINT")
        st.markdown(
            '<div class="approval-card"><strong>💬 ORBIT CẦN THÊM THÔNG TIN</strong><br>'
            "Graph đã tạm dừng đúng tại node chờ người dùng. Bạn có thể đóng lớp minh họa, "
            "trả lời trong chat, rồi graph sẽ tự phân loại lại để chọn node tiếp theo.</div>",
            unsafe_allow_html=True,
        )
        st.info(latest_state.get("pending_question") or "Bạn có thể nói rõ hơn không?")
        if st.button("ẨN GRAPH & TRẢ LỜI TRONG CHAT", type="primary", width="stretch"):
            st.session_state.pop("queued_chat_request", None)
            st.rerun(scope="app")
        return

    if phase == "approval":
        status_slot.warning("ĐÃ TẠM DỪNG // CẦN CON NGƯỜI PHÊ DUYỆT")
        st.markdown(
            '<div class="approval-card"><strong>⚠ CHECKPOINT CỦA CON NGƯỜI</strong><br>'
            "Orbit đã chuẩn bị hành động có tác động thật nhưng chưa gọi công cụ. "
            "Hãy kiểm tra và ra quyết định ngay tại đây.</div>",
            unsafe_allow_html=True,
        )
        if latest_state.get("proposed_action"):
            st.warning(latest_state["proposed_action"])
        comment = st.text_input(
            "Ghi chú của người duyệt",
            value="Đã xác minh danh tính khách hàng và chính sách hoàn tiền",
            key=f"popup-note-{latest_state.get('thread_id', 'approval')}",
        )
        approve_col, reject_col = st.columns(2)
        approved = approve_col.button("✓ PHÊ DUYỆT & TIẾP TỤC", type="primary", width="stretch")
        rejected = reject_col.button("✕ TỪ CHỐI", width="stretch")
        if approved or rejected:
            result = _resume_chat_request(
                str(queued["chat_id"]),
                cast(int, queued["turn_index"]),
                approved,
                comment,
                on_state=render_state,
            )
            queued["phase"] = _interrupt_phase(result)
            st.rerun()
        return

    status_slot.success("NHIỆM VỤ HOÀN TẤT // ĐÃ LƯU TOÀN BỘ TRACE")
    if st.button("VỀ CUỘC HỘI THOẠI", type="primary", width="stretch"):
        st.session_state.pop("queued_chat_request", None)
        st.rerun(scope="app")


def _render_conversation(checkpointer_kind: str) -> None:
    """Render the chat-first product experience and its request-bound graphs."""
    chat_id, chat = _active_conversation()
    turns = cast(list[dict[str, object]], chat["turns"])
    st.markdown('<div class="chat-shell">', unsafe_allow_html=True)
    st.markdown(
        '<div class="chat-head"><div class="agent-lockup">'
        '<div class="agent-orb">O</div><div><div class="agent-name">Orbit</div>'
        '<div class="agent-status"><span class="online-dot"></span>Trợ lý hỗ trợ · trực tuyến</div>'
        '</div></div><div class="mode-chip">GRAPH · TRỰC TIẾP</div></div>',
        unsafe_allow_html=True,
    )
    _conversation_toolbar()
    if "queued_chat_request" in st.session_state:
        _live_workflow_dialog(checkpointer_kind)

    if not turns:
        _render_video_welcome()
        st.caption("Chọn một tình huống mẫu hoặc nhắn tự nhiên như một khách hàng")
        shortcuts = [
            ("Câu hỏi đơn giản", "Đặt lại mật khẩu", "↗"),
            ("Tra cứu bằng công cụ", "Theo dõi đơn hàng", "⌁"),
            ("Thiếu thông tin", "Yêu cầu mơ hồ", "?"),
            ("Hoàn tiền cần HITL", "Phê duyệt hoàn tiền", "◇"),
            ("Lỗi tạm thời và phục hồi", "Xem agent retry", "↻"),
            ("Hết lượt thử", "Chuyển dead letter", "×"),
        ]
        for row in (shortcuts[:3], shortcuts[3:]):
            columns = st.columns(3)
            for column, (preset_name, label, icon) in zip(columns, row, strict=True):
                if column.button(f"{icon}  {label}", key=f"shortcut-{chat_id}-{preset_name}"):
                    _queue_chat_request(str(PRESETS[preset_name]["query"]), preset_name)
                    st.rerun()

    for index, turn in enumerate(turns, start=1):
        query = str(turn["query"])
        state = cast(AgentState, turn["state"])
        with st.chat_message("user", avatar="🙂"):
            st.markdown(query)
        for clarification in cast(list[dict[str, str]], turn.get("clarifications", [])):
            with st.chat_message("assistant", avatar="🪐"):
                st.markdown(clarification["question"])
            with st.chat_message("user", avatar="🙂"):
                st.markdown(clarification["answer"])
        with st.chat_message("assistant", avatar="🪐"):
            _render_turn_activity(state, index, expanded=index == len(turns))
            answer = state.get("final_answer") or state.get("pending_question")
            if _is_clarification_interrupt(state) and answer:
                st.markdown(answer)
                st.caption("Workflow đang tạm dừng tại node chờ người dùng — hãy trả lời bên dưới.")
            elif answer and not _has_interrupt(state):
                st.markdown(answer)
            elif _has_interrupt(state):
                st.markdown("Hành động đang chờ phê duyệt trong bảng nhiệm vụ trực tiếp.")
            _render_inline_approval(state, f"{chat_id}-{index}", chat_id, index - 1)

    prompt = st.chat_input("Nhắn tin cho Orbit…", key=f"composer-{chat_id}")
    if prompt and prompt.strip():
        reply = prompt.strip()
        waiting_for_detail = bool(turns) and _is_clarification_interrupt(
            cast(AgentState, turns[-1]["state"])
        )
        if waiting_for_detail:
            if st.session_state["show_live_graph"]:
                st.session_state["queued_chat_request"] = {
                    "query": str(turns[-1]["query"]),
                    "phase": "resume_clarification",
                    "chat_id": chat_id,
                    "turn_index": len(turns) - 1,
                    "clarification_answer": reply,
                    "notes": [],
                }
            else:
                _resume_chat_clarification(chat_id, len(turns) - 1, reply)
        elif st.session_state["show_live_graph"]:
            _queue_chat_request(reply)
        else:
            _submit_chat_request(reply, checkpointer_kind)
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def _run_scenario_suite(checkpointer_kind: str) -> None:
    st.write(
        "Chạy bảy tình huống chấm điểm bằng LLM thật. Các hành động rủi ro dùng phê duyệt "
        "mô phỏng để bộ đánh giá có thể chạy tự động."
    )
    if st.button("Chạy toàn bộ tình huống", type="primary"):
        scenarios = load_scenarios(SAMPLE_SCENARIOS)
        graph = _cached_graph(checkpointer_kind)
        progress = st.progress(0, text="Đang khởi động bộ tình huống...")
        items: list[ScenarioMetric] = []
        history_observed = False
        for index, scenario in enumerate(scenarios, start=1):
            state = initial_state(scenario)
            state["thread_id"] = f"ui-suite-{scenario.id}-{uuid4().hex[:6]}"
            config: RunnableConfig = {"configurable": {"thread_id": state["thread_id"]}}
            result = _invoke(graph, state, config, real_hitl=False)
            items.append(
                metric_from_state(
                    cast(dict[str, object], result),
                    scenario.expected_route.value,
                    scenario.requires_approval,
                )
            )
            if checkpointer_kind == "sqlite":
                history_observed = history_observed or bool(
                    next(graph.get_state_history(config), None)
                )
            progress.progress(index / len(scenarios), text=f"Đã hoàn tất {scenario.id}")
        report = summarize_metrics(items, resume_success=history_observed)
        st.session_state["suite_report"] = report
        progress.empty()

    if "suite_report" not in st.session_state:
        st.caption("Chưa chạy bộ tình huống nào trong phiên UI này.")
        return

    report = cast(MetricsReport, st.session_state["suite_report"])
    cols = st.columns(6)
    cols[0].metric("Thành công", f"{report.success_rate:.0%}")
    cols[1].metric("Tình huống", report.total_scenarios)
    cols[2].metric("Retry", report.total_retries)
    cols[3].metric("Phê duyệt", report.total_interrupts)
    cols[4].metric("Lượt gọi LLM", report.total_llm_calls)
    cols[5].metric("Fallback", report.total_structured_fallbacks)

    rows = [item.model_dump() for item in report.scenario_metrics]
    st.dataframe(rows, width="stretch", hide_index=True)

    report_markdown = render_report(report)
    metrics_json = json.dumps(report.model_dump(), indent=2, ensure_ascii=False)
    download_cols = st.columns(3)
    download_cols[0].download_button(
        "Tải metrics.json",
        metrics_json,
        file_name="metrics.json",
        mime="application/json",
        width="stretch",
    )
    download_cols[1].download_button(
        "Tải report.md",
        report_markdown,
        file_name="lab_report.md",
        mime="text/markdown",
        width="stretch",
    )
    if download_cols[2].button("Lưu vào repository", width="stretch"):
        write_metrics(report, UI_METRICS)
        write_report(report, UI_REPORT)
        st.success(f"Đã lưu {UI_METRICS.name} và {UI_REPORT.name}")

    with st.expander("Xem trước báo cáo"):
        st.markdown(report_markdown)


def _render_trace_and_checkpoints() -> None:
    if "last_result" not in st.session_state:
        st.info("Hãy chạy một workflow trong tab Hội thoại để xem trace và checkpoint.")
        return
    state = cast(AgentState, st.session_state["last_result"])
    graph = cast(CompiledStateGraph, st.session_state["last_graph"])
    config = cast(RunnableConfig, st.session_state["last_config"])

    st.subheader("Phát lại quá trình thực thi")
    path = _execution_path(state)
    slider_col, replay_col = st.columns([4, 1])
    replay_step = slider_col.slider(
        "Bước workflow",
        min_value=0,
        max_value=len(path) - 1,
        value=len(path) - 1,
        format="Step %d",
        key=f"replay-{state.get('thread_id', 'unknown')}",
    )
    replay_clicked = replay_col.button("Phát hoạt ảnh", width="stretch")
    graph_placeholder = st.empty()
    caption_placeholder = st.empty()
    if replay_clicked:
        for step in range(len(path)):
            graph_placeholder.graphviz_chart(
                _dynamic_graph_dot(state, upto_step=step), width="stretch"
            )
            caption_placeholder.caption(f"Step {step}: {_path_caption(path, step)}")
            sleep(0.45)
    else:
        graph_placeholder.graphviz_chart(
            _dynamic_graph_dot(state, upto_step=replay_step), width="stretch"
        )
        caption_placeholder.caption(f"Step {replay_step}: {_path_caption(path, replay_step)}")

    st.subheader("Trace theo từng node")
    st.dataframe(_event_rows(state), width="stretch", hide_index=True)

    st.subheader("Lịch sử checkpoint")
    st.caption(f"Thread ID: {state.get('thread_id', 'không xác định')}")
    history = _history_rows(graph, config)
    if history:
        st.dataframe(history, width="stretch", hide_index=True)
        st.success(f"Đã khôi phục {len(history)} snapshot checkpoint từ thread hiện tại.")
    else:
        st.warning("Lần chạy này không có lịch sử checkpoint.")

    st.subheader("Bằng chứng audit chỉ-ghi-thêm")
    audit_cols = st.columns(4)
    audit_cols[0].metric("Events", len(state.get("events", [])))
    audit_cols[1].metric("Messages", len(state.get("messages", [])))
    audit_cols[2].metric("Tool results", len(state.get("tool_results", [])))
    audit_cols[3].metric("Errors", len(state.get("errors", [])))


def _render_architecture() -> None:
    st.graphviz_chart(GRAPH_DOT, width="stretch")
    left, right = st.columns(2)
    with left:
        st.subheader("Điều hướng có điều kiện")
        st.dataframe(
            [
                {"route": "simple", "path": "answer → finalize"},
                {"route": "tool", "path": "tool → evaluate → answer/retry"},
                {
                    "route": "missing_info",
                    "path": "clarify → wait_for_user → classify",
                },
                {"route": "risky", "path": "risky_action → approval → tool/clarify"},
                {"route": "error", "path": "retry → tool/dead_letter"},
            ],
            width="stretch",
            hide_index=True,
        )
    with right:
        st.subheader("Cách cập nhật state")
        st.dataframe(
            [
                {"fields": "messages, tool_results, errors, events", "reducer": "append"},
                {
                    "fields": "route, attempt, evaluation, approval, final_answer",
                    "reducer": "overwrite",
                },
            ],
            width="stretch",
            hide_index=True,
        )
    st.markdown(
        """
        <span class="pill">Phân loại có cấu trúc</span>
        <span class="pill">Retry có giới hạn</span>
        <span class="pill">Con người trong vòng lặp</span>
        <span class="pill">Lưu state bằng SQLite</span>
        <span class="pill">LLM retry + fallback</span>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Render the chat-first, multi-conversation live-demo product."""
    st.set_page_config(
        page_title="Orbit · Agent Support",
        page_icon="✦",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _apply_theme()
    checkpointer_kind = _workspace_defaults()
    chat_tab, suite_tab, trace_tab, architecture_tab = st.tabs(
        ["✦ Hội thoại", "◫ Đánh giá", "⌁ Phát lại", "◇ Bản đồ hệ thống"]
    )
    with chat_tab:
        _render_conversation(checkpointer_kind)
    with suite_tab:
        _run_scenario_suite(checkpointer_kind)
    with trace_tab:
        _render_trace_and_checkpoints()
    with architecture_tab:
        _render_architecture()


if __name__ == "__main__":
    main()
