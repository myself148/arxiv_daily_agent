from tools.arxiv_client import fetch_latest_cv_papers


def test_fetch_latest_cv_papers():
    """测试论文抓取工具是否正常返回数据"""
    papers = fetch_latest_cv_papers(max_results=3)

    # 断言 1：返回的列表长度必须小于等于我们请求的最大数量
    assert len(papers) <= 3
    assert len(papers) > 0, "没有抓取到任何论文，请检查网络或 API"

    # 断言 2：检查返回的数据结构是否完整
    first_paper = papers[0]
    expected_keys = ["title", "authors", "published_date", "summary", "entry_id", "pdf_url"]
    for key in expected_keys:
        assert key in first_paper, f"论文数据缺失关键字段: {key}"
        assert first_paper[key] is not None