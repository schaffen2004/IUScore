import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import linear_sum_assignment

from src.core.iuscore import IUScore
from src.utils.preprocess import get_information_units, normalize_iu


def parse_information_units(raw_text: str) -> list[str]:
    return [line.strip() for line in raw_text.splitlines() if line.strip()]


def trace_span_extraction(span_extracter, question: str, iu: str) -> dict:
    input_ids, attention_mask, tokens = span_extracter.encode_input(question, iu)
    q_start, q_end, a_start, a_end = span_extracter.get_boundaries(tokens)

    outputs = span_extracter.model(input_ids=input_ids, attention_mask=attention_mask)
    attentions = outputs.attentions

    attention_scores = span_extracter.cross_attention(attentions, q_start, q_end, a_start, a_end)
    grad_scores = span_extracter.gradient_importance(input_ids, attention_mask)
    ig_scores = span_extracter.integrated_gradients(input_ids, attention_mask)
    combined_scores = span_extracter.combine_scores(attention_scores, grad_scores, ig_scores)

    best_start, best_end = span_extracter.extract_best_span(combined_scores, a_start, a_end)
    best_tokens = tokens[best_start : best_end + 1]
    best_span_text = span_extracter.tokenizer.convert_tokens_to_string(best_tokens)

    rows = []
    best_start_local = best_start - a_start
    best_end_local = best_end - a_start

    for local_idx, idx in enumerate(range(a_start, a_end)):
        rows.append(
            {
                "token_index": local_idx,
                "token": tokens[idx],
                "cross_attention": float(attention_scores[idx].detach().cpu().item()),
                "gradient": float(grad_scores[idx].detach().cpu().item()),
                "integrated_gradients": float(ig_scores[idx].detach().cpu().item()),
                "combined_score": float(combined_scores[idx]),
                "in_best_span": best_start_local <= local_idx <= best_end_local,
            }
        )

    return {
        "best_span_text": best_span_text,
        "best_span_indices": (best_start_local, best_end_local),
        "token_table": pd.DataFrame(rows),
    }


def render_span_section(title: str, traces: list[dict]) -> None:
    st.subheader(title)
    if not traces:
        st.info("Không có IU hợp lệ ở nhóm này.")
        return

    for trace in traces:
        st.markdown(f"**IU**: `{trace['iu']}`")
        st.markdown(f"**Extracted span**: `{trace['best_span_text']}`")

        token_df = trace["token_table"]
        st.dataframe(
            token_df,
            use_container_width=True,
            hide_index=True,
        )

        score_chart = (
            alt.Chart(token_df)
            .mark_bar()
            .encode(
                x=alt.X("token:N", title="Token", sort=None),
                y=alt.Y("combined_score:Q", title="Combined score"),
                color=alt.condition(
                    alt.datum.in_best_span,
                    alt.value("#d62728"),
                    alt.value("#4c78a8"),
                ),
                tooltip=["token_index", "token", "combined_score", "in_best_span"],
            )
            .properties(height=220)
        )
        st.altair_chart(score_chart, use_container_width=True)

        span_mask_df = token_df.copy()
        span_mask_df["best_span_mask"] = span_mask_df["in_best_span"].astype(int)
        mask_chart = (
            alt.Chart(span_mask_df)
            .mark_bar()
            .encode(
                x=alt.X("token:N", title="Token", sort=None),
                y=alt.Y("best_span_mask:Q", title="Best span (1/0)"),
                color=alt.condition(
                    alt.datum.in_best_span,
                    alt.value("#d62728"),
                    alt.value("#bdbdbd"),
                ),
                tooltip=["token_index", "token", "best_span_mask"],
            )
            .properties(height=120)
        )
        st.altair_chart(mask_chart, use_container_width=True)
        st.divider()


def render_matching_details(sim_matrix: np.ndarray, gt_spans: list[str], ans_spans: list[str]) -> None:
    if sim_matrix.size == 0:
        st.info("Similarity matrix rỗng vì thiếu spans ở GT hoặc Answer.")
        return

    row_ind, col_ind = linear_sum_assignment(-sim_matrix)
    match_rows = []
    for r, c in zip(row_ind, col_ind):
        match_rows.append(
            {
                "gt_span": gt_spans[r],
                "answer_span": ans_spans[c],
                "similarity": float(sim_matrix[r, c]),
            }
        )
    st.dataframe(pd.DataFrame(match_rows), use_container_width=True, hide_index=True)


@st.cache_resource(show_spinner=True)
def load_iu_score(model_extraction: str, model_encoding: str) -> IUScore:
    return IUScore(model_extraction=model_extraction, model_encoding=model_encoding)


def main() -> None:
    st.set_page_config(page_title="IUScore Visualizer", layout="wide")
    st.title("IUScore Visualizer")
    st.caption("Visualize toàn bộ pipeline: preprocessing -> span extraction -> similarity -> metrics")

    with st.sidebar:
        st.header("Model Config")
        model_extraction = st.text_input("Extraction model", value="vinai/phobert-base")
        model_encoding = st.text_input("Encoding model", value="all-mpnet-base-v2")
        run_button = st.button("Run IUScore", type="primary", use_container_width=True)

    question = st.text_input(
        "Question",
        value="Cá nhân đăng ký Bồi dưỡng nghiệp vụ đăng kiểm viên tàu cá phải nộp những loại giấy tờ gì?",
    )

    col1, col2 = st.columns(2)
    with col1:
        raw_gt = st.text_area(
            "Ground-truth IU list (mỗi dòng một IU)",
            value="* Đơn đề nghị tham gia bồi dưỡng nghiệp vụ đăng kiểm viên tàu cá theo Mẫu số 01.ĐKV Phụ lục II ban hành kèm theo Thông tư này.",
            height=220,
        )
    with col2:
        raw_answer = st.text_area(
            "Answer IU list (mỗi dòng một IU)",
            value="* Cá nhân phải nộp bản sao văn bằng và chứng chỉ chuyên môn.",
            height=220,
        )

    if not run_button:
        st.info("Nhập dữ liệu và bấm `Run IUScore` để xem chi tiết từng bước.")
        return

    gt_list = parse_information_units(raw_gt)
    answer_list = parse_information_units(raw_answer)

    with st.spinner("Loading models và chạy pipeline..."):
        scorer = load_iu_score(model_extraction=model_extraction, model_encoding=model_encoding)

        answer_iu, gt_iu = get_information_units(answer_list, gt_list)
        normalized_answer_iu = normalize_iu(answer_iu)
        normalized_gt_iu = normalize_iu(gt_iu)

        answer_traces = [
            {
                "iu": iu,
                **trace_span_extraction(scorer.span_extracter, question, iu),
            }
            for iu in normalized_answer_iu
        ]
        gt_traces = [
            {
                "iu": iu,
                **trace_span_extraction(scorer.span_extracter, question, iu),
            }
            for iu in normalized_gt_iu
        ]

        ans_spans = [item["best_span_text"] for item in answer_traces]
        gt_spans = [item["best_span_text"] for item in gt_traces]

        sim_matrix = scorer.similarity_matrix(gt_spans, ans_spans) if gt_spans and ans_spans else np.array([])

        if sim_matrix.size == 0:
            precision = recall = f1 = uncertainty = 0.0
        else:
            precision, recall, f1, uncertainty = scorer.compute_metrics(sim_matrix)

    st.header("1) Preprocessing")
    p_col1, p_col2 = st.columns(2)
    with p_col1:
        st.markdown("**Filtered IU (Answer)**")
        st.write(answer_iu)
        st.markdown("**Normalized IU (Answer)**")
        st.write(normalized_answer_iu)
    with p_col2:
        st.markdown("**Filtered IU (GT)**")
        st.write(gt_iu)
        st.markdown("**Normalized IU (GT)**")
        st.write(normalized_gt_iu)

    st.header("2) Span Extraction")
    c1, c2 = st.columns(2)
    with c1:
        render_span_section("Answer IU -> Span", answer_traces)
    with c2:
        render_span_section("GT IU -> Span", gt_traces)

    st.header("3) Similarity Matrix")
    if sim_matrix.size == 0:
        st.info("Không thể tính similarity matrix vì thiếu span ở GT hoặc Answer.")
    else:
        sim_df = pd.DataFrame(
            sim_matrix,
            index=[f"GT_{i}" for i in range(len(gt_spans))],
            columns=[f"ANS_{j}" for j in range(len(ans_spans))],
        )
        st.dataframe(sim_df, use_container_width=True)
        st.subheader("Hungarian Matching")
        render_matching_details(sim_matrix, gt_spans, ans_spans)

    st.header("4) Final Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precision", f"{precision:.4f}")
    m2.metric("Recall", f"{recall:.4f}")
    m3.metric("F1", f"{f1:.4f}")
    m4.metric("Uncertainty", f"{uncertainty:.4f}")


if __name__ == "__main__":
    main()
