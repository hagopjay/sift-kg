"""Tests for --json output on CLI commands."""

import json

from typer.testing import CliRunner

from sift_kg.cli import app

runner = CliRunner()


class TestInfoJson:
    """Test sift info --json output."""

    def test_info_json_outputs_valid_json(self, tmp_dir, sample_extraction):
        """sift info --json produces valid JSON to stdout."""
        from sift_kg.graph.builder import build_graph

        kg = build_graph([sample_extraction], postprocess=False)
        kg.save(tmp_dir / "graph_data.json")

        result = runner.invoke(app, ["info", "--json", "-o", str(tmp_dir)])
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert "entities" in data
        assert "relations" in data
        assert "domain" in data

    def test_info_json_excludes_document_nodes(self, tmp_dir, sample_extraction):
        """Entity count in JSON excludes DOCUMENT nodes."""
        from sift_kg.graph.builder import build_graph

        kg = build_graph([sample_extraction], postprocess=False)
        kg.save(tmp_dir / "graph_data.json")

        result = runner.invoke(app, ["info", "--json", "-o", str(tmp_dir)])
        data = json.loads(result.stdout)
        # sample_extraction has 3 entities (Alice, Acme, New York) + 1 DOCUMENT
        # JSON should only count the 3 substantive entities
        assert data["entities"] == 3

    def test_info_json_omits_missing_files(self, tmp_dir, sample_extraction):
        """Fields for missing files (merge_proposals, etc) are omitted."""
        from sift_kg.graph.builder import build_graph

        kg = build_graph([sample_extraction], postprocess=False)
        kg.save(tmp_dir / "graph_data.json")

        result = runner.invoke(app, ["info", "--json", "-o", str(tmp_dir)])
        data = json.loads(result.stdout)
        assert "merge_proposals" not in data
        assert "relation_review" not in data
