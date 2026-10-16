"""Archive fallback: a skill moved into ``skills/.archive/`` by the curator stays loadable by name.

Local patch — contract under test (see tools/skills_tool._locate_archived_skill,
tools/skill_manager_tool._find_skill, tools/skill_usage._find_skill_dir):
  * archiving hides a skill from the prompt index, nothing more;
  * a by-name lookup still resolves it (so references to it do not break silently);
  * the live tree always wins over an archived copy;
  * only direct children of ``.archive`` resolve — ``.archive/skill-backups/<name>``
    holds historical copies whose names collide with live skills and must be ignored.
"""

import json
from unittest.mock import patch

import pytest

from agent.skill_utils import find_archived_skill
from tools.skills_tool import skill_view


def _write_skill(parent, name: str, body: str):
    d = parent / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: test fixture {name}\n---\n\n{body}\n", encoding="utf-8")
    return d


@pytest.fixture()
def skills_tree(tmp_path):
    skills = tmp_path / "skills"
    archive = skills / ".archive"
    _write_skill(archive, "cold-skill", "Archived body")
    _write_skill(archive, "hot-skill", "Archived copy body")
    _write_skill(skills, "hot-skill", "Live body")
    # Historical pre-refactor copy: same skill name, nested deeper in the archive.
    _write_skill(archive / "skill-backups", "cold-skill", "Historical copy body")
    return skills


class TestFindArchivedSkill:
    def test_resolves_direct_archive_child(self, skills_tree):
        md = find_archived_skill("cold-skill", root=skills_tree)
        assert md is not None
        assert md.parent.name == "cold-skill"
        assert md.parent.parent.name == ".archive"

    def test_nested_backup_copy_does_not_shadow_the_direct_child(self, skills_tree):
        nested = skills_tree / ".archive" / "skill-backups" / "cold-skill" / "SKILL.md"
        assert nested.is_file()
        md = find_archived_skill("cold-skill", root=skills_tree)
        assert md is not None and md != nested
        assert "skill-backups" not in md.parts

    def test_categorized_form_is_flattened(self, skills_tree):
        md = find_archived_skill("devops/cold-skill", root=skills_tree)
        assert md is not None and md.parent.name == "cold-skill"

    def test_rejects_empty_dot_and_traversal_names(self, skills_tree):
        for bad in ("", ".", "..", "a/b/c", "../hot-skill", "x\\y"):
            assert find_archived_skill(bad, root=skills_tree) is None

    def test_missing_archive_returns_none(self, tmp_path):
        assert find_archived_skill("anything", root=tmp_path) is None


class TestSkillViewArchiveFallback:
    def test_archived_skill_loads_by_name(self, skills_tree):
        with patch("tools.skills_tool.SKILLS_DIR", skills_tree):
            payload = json.loads(skill_view("cold-skill"))
        assert "Archived body" in json.dumps(payload, ensure_ascii=False)

    def test_live_skill_wins_over_archived_copy(self, skills_tree):
        with patch("tools.skills_tool.SKILLS_DIR", skills_tree):
            raw = skill_view("hot-skill")
        assert "Live body" in raw
        assert "Archived copy body" not in raw

    def test_unknown_name_still_reports_not_found(self, skills_tree):
        with patch("tools.skills_tool.SKILLS_DIR", skills_tree):
            raw = skill_view("definitely-absent-skill")
        assert "not found" in raw.lower()


class TestUsageResolutionSeesArchivedLoads:
    def test_find_skill_dir_resolves_archived_skill(self, skills_tree):
        from tools import skill_usage
        with patch.object(skill_usage, "_skills_dir", lambda: skills_tree):
            found = skill_usage._find_skill_dir("cold-skill")
        assert found is not None and found.name == "cold-skill"

    def test_find_skill_dir_still_prefers_live_tree(self, skills_tree):
        from tools import skill_usage
        with patch.object(skill_usage, "_skills_dir", lambda: skills_tree):
            found = skill_usage._find_skill_dir("hot-skill")
        assert found is not None and found.parent == skills_tree
