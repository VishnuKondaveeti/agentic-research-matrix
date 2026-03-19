"""
Neo4j knowledge graph for research paper relationships.
Stores papers, authors, and topics as nodes with citation and authorship edges.
"""

from typing import Optional

from config.settings import settings


class KnowledgeGraph:
    """
    Neo4j-backed knowledge graph for research relationships.
    
    Nodes: Paper, Author, Topic
    Relationships: CITES, AUTHORED_BY, RELATES_TO
    """

    def __init__(self):
        self._driver = None
        self._available = False
        self._init_driver()

    def _init_driver(self):
        """Try to connect to Neo4j. Gracefully degrade if unavailable."""
        try:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password),
            )
            # Test connection
            with self._driver.session() as session:
                session.run("RETURN 1")
            self._available = True
            print("[KnowledgeGraph] Connected to Neo4j")
        except Exception as e:
            print(f"[KnowledgeGraph] Neo4j unavailable: {e}")
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def add_paper(self, paper: dict) -> bool:
        """
        Add a paper node and its relationships.

        Paper dict should have: title, authors (list), abstract, published, source, categories
        """
        if not self._available:
            return False

        title = paper.get("title", "")
        if not title:
            return False

        try:
            with self._driver.session() as session:
                # Create paper node
                session.run(
                    """
                    MERGE (p:Paper {title: $title})
                    SET p.abstract = $abstract,
                        p.published = $published,
                        p.source = $source,
                        p.arxiv_id = $arxiv_id
                    """,
                    title=title,
                    abstract=paper.get("abstract", "")[:500],
                    published=paper.get("published", ""),
                    source=paper.get("source", ""),
                    arxiv_id=paper.get("arxiv_id", ""),
                )

                # Create author nodes and relationships
                for author in paper.get("authors", []):
                    if isinstance(author, str) and author:
                        session.run(
                            """
                            MERGE (a:Author {name: $name})
                            WITH a
                            MATCH (p:Paper {title: $title})
                            MERGE (a)-[:AUTHORED]->(p)
                            """,
                            name=author,
                            title=title,
                        )

                # Create topic nodes from categories
                categories = paper.get("categories", [])
                if isinstance(categories, str):
                    categories = [categories]
                for topic in categories:
                    if topic:
                        session.run(
                            """
                            MERGE (t:Topic {name: $topic})
                            WITH t
                            MATCH (p:Paper {title: $title})
                            MERGE (p)-[:RELATES_TO]->(t)
                            """,
                            topic=topic,
                            title=title,
                        )

            return True
        except Exception as e:
            print(f"[KnowledgeGraph] Error adding paper: {e}")
            return False

    def add_citation(self, citing_title: str, cited_title: str) -> bool:
        """Add a citation relationship between two papers."""
        if not self._available:
            return False

        try:
            with self._driver.session() as session:
                session.run(
                    """
                    MATCH (p1:Paper {title: $citing})
                    MATCH (p2:Paper {title: $cited})
                    MERGE (p1)-[:CITES]->(p2)
                    """,
                    citing=citing_title,
                    cited=cited_title,
                )
            return True
        except Exception as e:
            print(f"[KnowledgeGraph] Error adding citation: {e}")
            return False

    def get_influential_papers(self, limit: int = 10) -> list[dict]:
        """Get most-cited papers."""
        if not self._available:
            return []

        try:
            with self._driver.session() as session:
                result = session.run(
                    """
                    MATCH (p:Paper)
                    OPTIONAL MATCH (p)<-[:CITES]-(citing:Paper)
                    WITH p, count(citing) AS citations
                    ORDER BY citations DESC
                    LIMIT $limit
                    RETURN p.title AS title, p.published AS published,
                           p.source AS source, citations
                    """,
                    limit=limit,
                )
                return [dict(record) for record in result]
        except Exception as e:
            print(f"[KnowledgeGraph] Query failed: {e}")
            return []

    def get_author_papers(self, author_name: str) -> list[dict]:
        """Get papers by a specific author."""
        if not self._available:
            return []

        try:
            with self._driver.session() as session:
                result = session.run(
                    """
                    MATCH (a:Author {name: $name})-[:AUTHORED]->(p:Paper)
                    RETURN p.title AS title, p.published AS published, p.source AS source
                    ORDER BY p.published DESC
                    """,
                    name=author_name,
                )
                return [dict(record) for record in result]
        except Exception as e:
            return []

    def get_related_papers(self, title: str, limit: int = 5) -> list[dict]:
        """Find papers related by shared topics or authors."""
        if not self._available:
            return []

        try:
            with self._driver.session() as session:
                result = session.run(
                    """
                    MATCH (p1:Paper {title: $title})-[:RELATES_TO]->(t:Topic)<-[:RELATES_TO]-(p2:Paper)
                    WHERE p1 <> p2
                    WITH p2, count(t) AS shared_topics
                    ORDER BY shared_topics DESC
                    LIMIT $limit
                    RETURN p2.title AS title, p2.published AS published, shared_topics
                    """,
                    title=title,
                    limit=limit,
                )
                return [dict(record) for record in result]
        except Exception as e:
            return []

    def get_graph_stats(self) -> dict:
        """Get graph statistics."""
        if not self._available:
            return {"available": False}

        try:
            with self._driver.session() as session:
                papers = session.run("MATCH (p:Paper) RETURN count(p) AS count").single()["count"]
                authors = session.run("MATCH (a:Author) RETURN count(a) AS count").single()["count"]
                topics = session.run("MATCH (t:Topic) RETURN count(t) AS count").single()["count"]
                citations = session.run("MATCH ()-[r:CITES]->() RETURN count(r) AS count").single()["count"]

                return {
                    "available": True,
                    "papers": papers,
                    "authors": authors,
                    "topics": topics,
                    "citations": citations,
                }
        except Exception as e:
            return {"available": False, "error": str(e)}

    def get_full_graph(self, limit: int = 100) -> dict:
        """Get nodes and links for visualization."""
        if not self._available:
            return {"nodes": [], "links": []}

        try:
            with self._driver.session() as session:
                # Fetch Papers and Authors
                result = session.run(
                    """
                    MATCH (n)
                    WHERE n:Paper OR n:Author
                    WITH n LIMIT $limit
                    OPTIONAL MATCH (n)-[r]->(m)
                    WHERE m:Paper OR m:Author
                    RETURN n, r, m
                    """,
                    limit=limit
                )
                
                nodes = {}
                links = []
                
                for record in result:
                    n = record["n"]
                    if n:
                        nodes[n.element_id] = {
                            "id": n.get("title") or n.get("name"),
                            "group": 1 if "Paper" in n.labels else 2,
                            "label": n.get("title") or n.get("name")
                        }
                    
                    m = record["m"]
                    if m:
                        nodes[m.element_id] = {
                            "id": m.get("title") or m.get("name"),
                            "group": 1 if "Paper" in m.labels else 2,
                            "label": m.get("title") or m.get("name")
                        }
                    
                    r = record["r"]
                    if r:
                        links.append({
                            "source": nodes[n.element_id]["id"],
                            "target": nodes[m.element_id]["id"],
                            "value": 1
                        })
                
                return {
                    "nodes": list(nodes.values()),
                    "links": links
                }
        except Exception as e:
            return {"nodes": [], "links": [], "error": str(e)}

    def close(self):
        """Close the Neo4j driver."""
        if self._driver:
            self._driver.close()
