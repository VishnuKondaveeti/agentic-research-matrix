/**
 * A.I.R.A - Frontend Logic (Vanilla JS)
 * Handles SPA navigation, API requests, and WebSockets (simulated for now, to be mapped to fastAPI).
 */

const API_BASE = "http://localhost:8000/api";
const WS_BASE = "ws://localhost:8000/api/ws/logs";

// WebSocket Global
let logSocket = null;

// LLM Provider Global
let currentProvider = localStorage.getItem('llm_provider') || 'gemini';


// DOM Elements
const navItems = document.querySelectorAll('.nav-item');
const views = document.querySelectorAll('.view');
const topTitle = document.getElementById('top-title');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Nav Click Listeners
    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const viewName = item.getAttribute('data-view');
            switchView(viewName);
        });
    });

    // Research View bindings
    document.getElementById('btn-search-init').addEventListener('click', startResearchCascade);
    document.getElementById('btn-hitl-approve').addEventListener('click', approveHitlProcessor);
    document.getElementById('btn-hitl-cancel').addEventListener('click', () => {
        document.getElementById('hitl-panel').style.display = 'none';
        logToConsole("research", "Process aborted by human authorization.", "log-error");
    });

    // Chat bindings
    document.getElementById('btn-chat-send').addEventListener('click', sendChatMessage);
    document.getElementById('chat-input').addEventListener('keypress', (e) => {
        if(e.key === 'Enter') sendChatMessage();
    });

    // Analytics Tabs bindings
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            switchTab(btn.getAttribute('data-target'));
        });



    });
    
    // Synthesis View bindings
    const btnGenReport = document.getElementById('btn-generate-report');
    if (btnGenReport) btnGenReport.addEventListener('click', generateSynthesisReport);
    
    // Slider label updates
    const reportK = document.getElementById('report-k');
    const reportKVal = document.getElementById('report-k-val');
    if(reportK && reportKVal) {
        reportK.addEventListener('input', () => {
            reportKVal.textContent = `${reportK.value} nodes`;
        });
    }

    // LLM Provider bindings
    const providerSelect = document.getElementById('provider-select');
    const modelBadge = document.getElementById('active-model-badge');
    
    if(providerSelect) {
        providerSelect.value = currentProvider;
        if(modelBadge) modelBadge.textContent = currentProvider.charAt(0).toUpperCase() + currentProvider.slice(1);
        
        providerSelect.addEventListener('change', (e) => {
            currentProvider = e.target.value;
            localStorage.setItem('llm_provider', currentProvider);
            if(modelBadge) {
                modelBadge.textContent = currentProvider.charAt(0).toUpperCase() + currentProvider.slice(1);
                // Subtle pulse effect
                modelBadge.style.animation = 'none';
                modelBadge.offsetHeight;
                modelBadge.style.animation = 'pulse 1s linear';
            }
            logToConsole("research", `LLM Provider switched to: ${currentProvider}`, "log-step");
        });
    }

    // Initial fetch to check db stats
    fetchStats();
    
    // Connect WebSockets
    connectWebSockets();

    // Fade out init overlay
    setTimeout(() => {
        const overlay = document.getElementById('init-overlay');
        if(overlay) overlay.classList.add('fade-out');
    }, 1500);
});


// === WebSocket Connection ===
function connectWebSockets() {
    logSocket = new WebSocket(WS_BASE);
    
    logSocket.onopen = () => {
        console.log("WebSocket connected matrix.");
    };

    logSocket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === "log") {
            // Determine which console to log to based on context or just log to both/active
            const activeView = document.querySelector('.view.active').id;
            let consoleType = "research"; // default
            if (activeView === "view-reports") consoleType = "reports";
            
            const levelClass = data.level === "error" ? "log-error" : 
                               data.level === "warning" ? "log-warning" : 
                               data.agent === "OrchestratorAgent" ? "log-step" : "log-line";
            
            logToConsole(consoleType, `${data.message}`, levelClass);
        } else if (data.type === "debate") {
            logToWarRoom(data.agent, data.message);
        }
    };

    logSocket.onclose = () => {
        console.warn("WebSocket disconnected. Retrying in 5s...");
        setTimeout(connectWebSockets, 5000);
    };
}


// === Navigation ===
function switchView(viewName) {
    views.forEach(v => v.classList.remove('active'));
    navItems.forEach(i => i.classList.remove('active'));
    
    const activeView = document.getElementById(`view-${viewName}`);
    const activeNav = document.querySelector(`.nav-item[data-view="${viewName}"]`);
    
    if (activeView && activeNav) {
        activeView.classList.add('active');
        activeNav.classList.add('active');
        
        // Update breadcrumbs
        const viewTitle = activeNav.querySelector('span').textContent;
        topTitle.innerHTML = `Research / <span>${viewTitle}</span>`;
        
        // Lazy load specific charts if navigating to them
        if(viewName === 'analytics') {
            switchTab('tab-graph'); // Default tab
        }
    }
}

function switchTab(tabId) {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    
    const btn = document.querySelector(`.tab-btn[data-target="${tabId}"]`);
    const content = document.getElementById(tabId);
    
    if(btn && content) {
        btn.classList.add('active');
        content.classList.add('active');
        
        // Initialize based on tab
        if(tabId === 'tab-graph') initGraph();
        if(tabId === 'tab-3d') initVectorSpace();
        if(tabId === 'tab-trends') initTrends();
        if(tabId === 'tab-leaderboard') syncLeaderboard();
    }
}



// === API Interactors ===

async function fetchStats() {
    try {
        const res = await fetch(`${API_BASE}/health`);
        const data = await res.json();
        document.querySelector('#sidebar-db-stats .highlight').textContent = data.vector_db_documents || "0";
    } catch(e) {
        console.warn("Backend not running yet, using local mocks.");
    }
}

// --- Research / Ingestion --- //

function logToConsole(consoleType, msg, typeClass="log-line") {
    const out = document.getElementById(`console-output-${consoleType}`);
    const div = document.createElement('div');
    div.className = typeClass;
    
    const time = new Date().toLocaleTimeString();
    div.innerHTML = `<span style="color:#5c6fff">[${time}]</span> ${msg}`;
    
    out.appendChild(div);
    out.scrollTop = out.scrollHeight;
}

function logToWarRoom(agentId, content) {
    const container = document.getElementById('war-room-container');
    container.style.display = 'block';
    
    const log = document.getElementById('war-debate-log');
    const div = document.createElement('div');
    div.className = `debate-message ${agentId.toLowerCase()}`;
    
    const agentName = agentId.charAt(0).toUpperCase() + agentId.slice(1);
    div.innerHTML = `<span class="author">${agentName}:</span> <span class="content">${content}</span>`;
    
    log.appendChild(div);
    log.scrollTop = log.scrollHeight;
    
    // Highlight speaking agent
    document.querySelectorAll('.war-agent-box').forEach(b => b.classList.remove('speaking'));
    const speakerBox = document.getElementById(`war-agent-${agentId.toLowerCase()}`);
    if(speakerBox) {
        speakerBox.classList.add('speaking');
        speakerBox.querySelector('.war-thought').innerText = content;
    }
}

let mockFoundPapers = [];

let insightInterval = null;

function startInsightStream() {
    const streamer = document.getElementById('insight-streamer');
    const facts = [
        "Analyzing multidimensional vector manifolds...",
        "Traversing Neo4j knowledge topology...",
        "Did you know? Transformers were first proposed in 2017.",
        "Synthesizing elite research insights...",
        "Cross-referencing global paper indices...",
        "Optimizing RAG retrieval pathways...",
        "Detecting emerging patterns in AI ethics...",
        "Mapping researcher influence via citation graphs..."
    ];
    let idx = 0;
    insightInterval = setInterval(() => {
        streamer.innerHTML = `<div class="fact-text">${facts[idx % facts.length]}</div>`;
        idx++;
    }, 3000);
}

function stopInsightStream() {
    if(insightInterval) clearInterval(insightInterval);
}

async function startResearchCascade() {
    const query = document.getElementById('research-query').value;
    if(!query) return alert("Enter a target topic");
    
    // UI Activation
    document.getElementById('neural-pulse-loader').style.display = 'flex';
    document.getElementById('hitl-panel').style.display = 'none';
    document.getElementById('agent-console-research').style.display = 'none';
    
    startInsightStream();

    // Call the search API
    try {
        const res = await fetch(`${API_BASE}/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: query,
                max_papers: parseInt(document.getElementById('research-max').value),
                complexity: document.getElementById('research-complexity').value,
                sources: ["arxiv"],
                download: false,
                process: false,
                llm_provider: currentProvider
            })
        });
        const data = await res.json();
        
        stopInsightStream();
        document.getElementById('neural-pulse-loader').style.display = 'none';

        if (data.status === "success") {
            showHITLPanel(query, data.papers);
            document.getElementById('agent-console-research').style.display = 'block';
        } else {
            logToConsole("research", `Search failed: ${data.message}`, "log-error");
        }
    } catch (e) {
        stopInsightStream();
        document.getElementById('neural-pulse-loader').style.display = 'none';
        logToConsole("research", `Network error: ${e.message}`, "log-error");
    }
}


function showHITLPanel(query, papers) {
    document.getElementById('hitl-panel').style.display = 'block';
    const list = document.getElementById('hitl-paper-list');
    list.innerHTML = "";
    
    document.getElementById('hitl-count').textContent = papers.length;

    papers.forEach(p => {
        const item = document.createElement('div');
        item.style.padding = "10px";
        item.style.borderBottom = "1px solid rgba(255,255,255,0.05)";
        item.style.display = "flex";
        item.style.alignItems = "center";
        item.style.gap = "12px";

        const scoreColor = p.influence_score > 80 ? "#4ade80" : (p.influence_score > 60 ? "#fbbf24" : "#9ca3af");

        item.innerHTML = `
            <label class="cb-container" style="flex-shrink:0;">
                <input type="checkbox" checked data-pid="${p.id || p.title}" data-score="${p.influence_score || 0}" data-title="${p.title}">
                <span class="checkmark"></span>
            </label>
            <div style="flex-grow:1;">
                <div style="font-weight:600; font-size:0.95rem;">${p.title}</div>
                <div style="font-size:0.8rem; color:var(--text-muted);"><span class="source-pill">${p.source}</span> ${p.published || ""}</div>
            </div>
            <div class="influence-score-pill" style="border-color:${scoreColor}; color:${scoreColor}">
                <i class="fa-solid fa-fire"></i> ${p.influence_score || 65}
            </div>
        `;
        list.appendChild(item);
    });
}


// --- Visualization Logic --- //

async function initTrends() {
    const timelineContainer = document.getElementById('timeline-chart');
    if(!timelineContainer) return;

    try {
        // Render timeline simple bar chart with premium styling
        const trace = {
            x: ["2022", "2023", "2024", "2025", "2026"],
            y: [12, 18, 25, 42, 65],
            type: 'bar',
            marker: { color: '#5c6fff', corners: 10 }
        };
        const layout = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#6e6e73', family: 'Inter' },
            margin: { l: 40, r: 20, t: 40, b: 40 },
            xaxis: { gridcolor: 'rgba(0,0,0,0.05)' },
            yaxis: { gridcolor: 'rgba(0,0,0,0.05)' },
            showlegend: false
        };
        Plotly.newPlot('timeline-chart', [trace], layout, {responsive: true, displayModeBar: false});
        
        // Update the global metrics counts (simulated)
        document.getElementById('global-top-researcher').innerText = "Yann LeCun";
        document.getElementById('global-trending-topic').innerText = "World Models + RAG";
        document.getElementById('global-most-discussed').innerText = "JEPA Architecture";
        
    } catch (e) {
        console.error("Trends init failed", e);
    }
}




async function approveHitlProcessor() {
    document.getElementById('hitl-panel').style.display = 'none';
    
    // Get selected papers
    const selected = Array.from(document.querySelectorAll('#hitl-paper-list input:checked'))
                          .map(el => el.getAttribute('data-pid'));
    
    logToConsole("research", "Human authorization received. Resuming Subspace Cascade...", "log-step");
    
    try {
        const style = document.getElementById('research-style').value;
        const res = await fetch(`${API_BASE}/ingest`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: document.getElementById('research-query').value,
                paper_ids: selected,
                style: style,
                llm_provider: currentProvider
            })
        });
        const data = await res.json();
        
        if (data.status === "success" || data.stages) {
            logToConsole("research", "Cascade completed successfully.", "log-success");
            fetchStats();
            
            // Handle Elite Insights (Blueprint & Debate)
            if (data.stages) {
                const stages = data.stages;
                
                // Show in Synthesis View automatically
                const blueprint = stages.blueprint?.blueprint || "No blueprint generated.";
                const debate = stages.debate?.debate || "No debate generated.";
                const hypotheses = stages.hypotheses?.hypotheses || "No hypotheses generated.";
                
                document.getElementById('blueprint-content').innerHTML = formatMarkdown(blueprint);
                document.getElementById('debate-content').innerHTML = formatMarkdown(debate);
                document.getElementById('hypothesis-content').innerHTML = formatMarkdown(hypotheses);
                document.getElementById('elite-insights').style.display = 'block';
                
                // If there's an analysis/report, show it too
                if (stages.analysis) {
                    const reportContent = document.getElementById('report-content');
                    const emptyState = document.getElementById('report-empty');
                    reportContent.innerHTML = formatMarkdown(stages.analysis.analysis || stages.analysis.summary);
                    reportContent.style.display = 'block';
                    emptyState.style.display = 'none';
                    document.getElementById('report-actions').style.display = 'flex';
                }

                // Switch to Reports view to show results
                setTimeout(() => {
                    switchView('reports');
                    logToConsole("reports", "Elite insights synthesized and ready for review.", "log-success");
                }, 1500);
            }

            // Show Gap Alert if advisor ran
            if (data.stages?.analysis) {
                const gapContainer = document.getElementById('gap-alert-container');
                if(gapContainer) gapContainer.style.display = 'block';
            }
        } else {
            logToConsole("research", `Ingestion failed: ${data.message || "Unknown error"}`, "log-error");
        }
    } catch (e) {
        logToConsole("research", `Network error: ${e.message}`, "log-error");
    }
}
async function simulateWarRoomDebate(topic) {
    const container = document.getElementById('war-room-container');
    if(!container) return;
    container.style.display = 'block';
    
    // Clear previous debate log
    document.getElementById('war-debate-log').innerHTML = "";

    const debateSteps = [
        { agent: "critic", msg: `The methodology in the latest papers for "${topic}" seems over-reliant on synthetic benchmarks. We need more real-world validation.` },
        { agent: "optimist", msg: "While synthetic data is a concern, the scaling laws shown in these clusters suggest massive efficiency gains in zero-shot tasks!" },
        { agent: "advisor", msg: "I recommend a cross-domain verification. We should pivot our next inquiry towards the research gap in robotics integration." },
        { agent: "critic", msg: "Agreed. But only if we can verify the latency constraints in those robotics environments." }
    ];

    let step = 0;
    // Log first step immediately
    logToWarRoom(debateSteps[step].agent, debateSteps[step].msg);
    step++;

    const interval = setInterval(() => {
        if(step < debateSteps.length) {
            logToWarRoom(debateSteps[step].agent, debateSteps[step].msg);
            step++;
        } else {
            clearInterval(interval);
            setTimeout(() => {
                document.querySelectorAll('.war-agent-box').forEach(b => b.classList.remove('speaking'));
            }, 2000);
        }
    }, 4000);
}



async function generateSynthesisReport() {
    const topic = document.getElementById('report-topic').value;
    const complexity = document.getElementById('report-complexity').value;
    const style = document.getElementById('report-type').value;
    if(!topic) return alert("Enter a synthesis topic");

    const reporterOutput = document.getElementById('report-content');
    const emptyState = document.getElementById('report-empty');
    const agentConsole = document.getElementById('agent-console-reports');
    
    reporterOutput.style.display = 'none';
    emptyState.style.display = 'none';
    agentConsole.style.display = 'block';
    logToConsole("reports", `Initializing Synthesis Cascade for: ${topic}`, "log-step");

    try {
        const res = await fetch(`${API_BASE}/report`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                topic: topic,
                complexity: complexity,
                style: style,
                top_k: parseInt(document.getElementById('report-k').value),
                llm_provider: currentProvider
            })
        });
        const data = await res.json();
        
        if (data.status === "success") {
            reporterOutput.innerHTML = formatMarkdown(data.report);
            reporterOutput.style.display = 'block';
            
            const eliteInsights = document.getElementById('elite-insights');
            let hasEliteContent = false;

            // Helper to show/hide sections
            const toggleSection = (id, contentId, content, condition) => {
                const container = document.getElementById(id);
                const target = document.getElementById(contentId);
                if (container && target && condition && content) {
                    target.innerHTML = formatMarkdown(content);
                    container.style.display = 'block';
                    hasEliteContent = true;
                } else if (container) {
                    container.style.display = 'none';
                }
            };

            // Handle individual insights
            if (data.results) {
                toggleSection('elite-blueprint', 'blueprint-content', data.results.blueprint?.blueprint, data.results.blueprint);
                toggleSection('elite-validation', 'validation-content', data.results.validation?.environment_spec, data.results.validation);
                toggleSection('elite-debate', 'debate-content', data.results.debate?.debate, data.results.debate);
                toggleSection('elite-trend', 'trend-content', data.results.trends?.forecast, data.results.trends);
                toggleSection('elite-hypothesis', 'hypothesis-content', data.results.hypotheses?.hypotheses, data.results.hypotheses);
                toggleSection('elite-podcast', 'podcast-content', data.results.podcast?.script, data.results.podcast);
            }

            if (hasEliteContent) {
                eliteInsights.style.display = 'block';
            } else {
                eliteInsights.style.display = 'none';
            }

            document.getElementById('report-actions').style.display = 'flex';
            logToConsole("reports", "Synthesis completed.", "log-success");
        } else {
            logToConsole("reports", `Synthesis failed: ${data.message}`, "log-error");
        }
    } catch(e) {
        logToConsole("reports", `Network error: ${e.message}`, "log-error");
    }
}


// --- Chat --- //

async function sendChatMessage() {
    const input = document.getElementById('chat-input');
    const msg = input.value.trim();
    const complexity = document.getElementById('chat-complexity').value;
    if(!msg) return;

    appendMessage('user', msg);
    input.value = "";
    
    const thoughtStream = document.getElementById('chat-thought-stream');
    thoughtStream.style.display = 'block';
    
    const thoughts = [
        "Analyzing intent...",
        "Querying ChromaDB Top-K = 5...",
        "Traversing Neo4j adjacent nodes...",
        "Applying complexity constraints..."
    ];
    
    let t_idx = 0;
    const txtSpan = thoughtStream.querySelector('.thought-text');
    txtSpan.textContent = thoughts[0];
    
    const intv = setInterval(() => {
        t_idx++;
        if(t_idx < thoughts.length) {
            txtSpan.textContent = thoughts[t_idx];
        } else {
            // Keep the last thought until request finishes
        }
    }, 500);

    try {
        const res = await fetch(`${API_BASE}/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: msg,
                complexity: complexity,
                top_k: 5,
                llm_provider: currentProvider
            })
        });
        const data = await res.json();
        
        clearInterval(intv);
        thoughtStream.style.display = 'none';

        if (data.answer) {
            appendMessage('assistant', data.answer);
            if (data.sources && data.sources.length > 0) {
                const sourceList = data.sources.map(s => `[${s.title}]`).join(', ');
                const sourcesDiv = document.createElement('div');
                sourcesDiv.className = "source-list-mini";
                sourcesDiv.style.fontSize = "0.75rem";
                sourcesDiv.style.opacity = "0.6";
                sourcesDiv.style.marginTop = "8px";
                sourcesDiv.innerHTML = `<strong>Sources:</strong> ${sourceList}`;
                document.querySelector('.message.assistant:last-child .content').appendChild(sourcesDiv);
            }
        } else {
            appendMessage('assistant', "Error: " + (data.message || "Unknown error"));
        }
    } catch(e) {
        clearInterval(intv);
        thoughtStream.style.display = 'none';
        appendMessage('assistant', "Network Error: Could not reach the agent cluster. Ensure FastAPI is running.");
    }
}

function appendMessage(role, content) {
    const history = document.getElementById('chat-history');
    const div = document.createElement('div');
    div.className = `message ${role}`;
    
    const icon = role === 'user' ? '<i class="fa-solid fa-user"></i>' : '<i class="fa-solid fa-brain"></i>';
    
    div.innerHTML = `
        <div class="avatar">${icon}</div>
        <div class="content">${content}</div>
    `;
    history.appendChild(div);
    history.scrollTop = history.scrollHeight;
}


// --- Visualizations (D3 & Plotly Mocks) --- //

let network = null;

async function initGraph() {
    const container = document.getElementById('d3-graph-canvas');
    if(!container || !window.vis) return;
    
    try {
        const res = await fetch(`${API_BASE}/analytics/graph`);
        const graphData = await res.json();
        
        if (graphData.status !== "success") throw new Error("Graph sync failed");

        const nodes = new vis.DataSet(graphData.graph.nodes);
        const edges = new vis.DataSet(graphData.graph.edges);

        const data = { nodes: nodes, edges: edges };
        const options = {
            nodes: {
                shape: 'dot',
                size: 25,
                font: { size: 14, color: '#f8f8f2', face: 'Inter' },
                borderWidth: 2,
                shadow: true
            },
            edges: {
                width: 2,
                color: { inherit: 'from' },
                smooth: { type: 'continuous' }
            },
            groups: {
                core: { color: {background: '#5c6fff', border: '#3a4dfa'}, size: 30 },
                agent: { color: {background: '#ff5c8d', border: '#fa3a74'} },
                storage: { color: {background: '#00f0ff', border: '#00c8d7'} },
                source: { color: {background: '#00ffaa', border: '#00d78f'} }
            },
            physics: {
                forceAtlas2Based: { gravitationalConstant: -100, centralGravity: 0.01, springLength: 150 },
                maxVelocity: 50,
                solver: 'forceAtlas2Based',
                timestep: 0.35,
                stabilization: { iterations: 150 }
            }
        };

        if (network) network.destroy();
        network = new vis.Network(container, data, options);
        
        // Interaction
        network.on("click", function (params) {
            if (params.nodes.length > 0) {
                const nodeId = params.nodes[0];
                const node = nodes.get(nodeId);
                logToConsole("research", `Investigating node: ${node.label} - ${node.title}`, "log-step");
            }
        });
    } catch (e) {
        console.error("Graph init failed", e);
    }
}
async function initVectorSpace() {
    if(!window.Plotly) return;
    const container = document.getElementById('plotly-3d-canvas');
    if(!container) return;

    try {
        const res = await fetch(`${API_BASE}/analytics/embeddings?dimensions=3`);
        const vectorData = await res.json();

        if (vectorData.status !== "success") throw new Error("Vector sync failed");

        const data = [{
            x: vectorData.points.map(p => p.x),
            y: vectorData.points.map(p => p.y),
            z: vectorData.points.map(p => p.z),
            text: vectorData.points.map(p => p.title),
            mode: 'markers',
            marker: { 
                size: 4, 
                color: '#5c6fff', 
                opacity: 0.8,
                line: { width: 0.5, color: 'rgba(255,255,255,0.1)' }
            },
            type: 'scatter3d'
        }];

        const layout = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            margin: {l: 0, r: 0, b: 0, t: 0},
            scene: {
                xaxis: {showgrid: false, zeroline: false, showbackground: false, visible: false},
                yaxis: {showgrid: false, zeroline: false, showbackground: false, visible: false},
                zaxis: {showgrid: false, zeroline: false, showbackground: false, visible: false},
                camera: {
                    eye: {x: 1.5, y: 1.5, z: 1.5}
                }
            },
            font: {color: "#6e6e73"}
        };

        Plotly.newPlot('plotly-3d-canvas', data, layout, {displayModeBar: false, responsive: true});
    } catch (e) {
        console.error("Vector Space init failed", e);
    }
}

async function searchGlobalTrends() {
    const query = document.getElementById('trends-search-input').value;
    if(!query) return alert("Enter a name or topic");

    logToConsole("research", `Querying Global Trends for: ${query}...`, "log-step");
    
    try {
        const res = await fetch(`${API_BASE}/trends/${query}`);
        const data = await res.json();
        
        if (data.error) {
            return logToConsole("research", `Trends Error: ${data.error}`, "log-warning");
        }

        // Update Metrics
        if (data.top_authors && data.top_authors.length > 0) {
            document.getElementById('global-top-researcher').textContent = data.top_authors[0].name;
        } else {
            document.getElementById('global-top-researcher').textContent = query;
        }

        if (data.top_keywords && data.top_keywords.length > 0) {
            document.getElementById('global-trending-topic').textContent = data.top_keywords[0].keyword;
        }
        if (data.papers && data.papers.length > 0) {
            // Find a paper with a title
            const validPaper = data.papers.find(p => p.title && p.title !== "Untitled");
            if (validPaper) {
                document.getElementById('global-most-discussed').textContent = validPaper.title;
            }
        }

        // Update Chart
        if (data.timeline && data.timeline.length > 0) {
            const x = data.timeline.map(t => t.period);
            const y = data.timeline.map(t => t.count);
            
            const trace = {
                x: x,
                y: y,
                type: 'bar',
                marker: { color: '#5c6fff' }
            };
            const layout = {
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: { color: '#6e6e73', family: 'Inter' },
                margin: { l: 40, r: 20, t: 40, b: 40 },
                xaxis: { gridcolor: 'rgba(0,0,0,0.05)' },
                yaxis: { gridcolor: 'rgba(0,0,0,0.05)' },
                showlegend: false
            };
            Plotly.newPlot('timeline-chart', [trace], layout, {responsive: true, displayModeBar: false});
        }

        // Refresh Graph context
        if(window.initGraph) initGraph();
        logToConsole("research", `Analytics synchronized for: ${query}`, "log-success");

    } catch (e) {
        logToConsole("research", `Trends Network Error: ${e.message}`, "log-error");
    }
}

function selectAllPapers(checked) {
    document.querySelectorAll('#hitl-paper-list input').forEach(cb => {
        cb.checked = checked;
    });
}

function filterHighImpact() {
    document.querySelectorAll('#hitl-paper-list input').forEach(cb => {
        const score = parseInt(cb.getAttribute('data-score'));
        cb.checked = score >= 80;
    });
}

function exportBibliography() {
    // Collect all titles from CHECKED checkboxes
    const checkboxes = Array.from(document.querySelectorAll('#hitl-paper-list input:checked'));
    const papers = checkboxes.map(cb => cb.getAttribute('data-title'));
    
    if(papers.length === 0) return alert("Search for papers first to generate a bibliography.");

    let bibtex = papers.map((title, i) => {
        return `@article{paper${i},\n  title={${title}},\n  author={AIRA Matrix Agents},\n  year={2026}\n}`;
    }).join("\n\n");

    const blob = new Blob([bibtex], {type: 'text/plain'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'aira_bibliography.bib';
    a.click();
    
    logToConsole("reports", "Bibliography exported successfully.", "log-success");
}


// --- Knowledge Dropzone ---

const dropzone = document.getElementById('knowledge-dropzone');
const dropzoneInput = document.getElementById('dropzone-input');

if (dropzone) {
    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('drag-over');
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('drag-over');
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('drag-over');
        const files = e.dataTransfer.files;
        if (files.length > 0) handleFileUpload(files);
    });

    dropzoneInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) handleFileUpload(e.target.files);
    });
}

async function handleFileUpload(files) {
    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
        if (files[i].type !== 'application/pdf') {
            alert("Only PDFs are allowed in the elite subspace.");
            return;
        }
        formData.append('file', files[i]);
    }

    // Toggle UI State
    document.querySelector('.dropzone-content').style.display = 'none';
    document.getElementById('dropzone-status').style.display = 'flex';
    
    logToConsole("research", `Ingesting ${files.length} local documents...`, "log-step");

    try {
        const res = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            body: formData
        });
        const data = await res.json();
        
        if (data.status === "success") {
            logToConsole("research", "Local ingestion complete. Knowledge integrated into vector space.", "log-success");
            
            // Show success summary
            document.getElementById('dropzone-status-text').innerHTML = 
                `<span style="color:#4ade80">Ingestion Successful!</span><br>${files.length} papers analyzed.`;
            
            setTimeout(() => {
                document.querySelector('.dropzone-content').style.display = 'block';
                document.getElementById('dropzone-status').style.display = 'none';
                document.getElementById('dropzone-status-text').textContent = "Ingesting local wisdom...";
            }, 5000);

            // Refresh graph to show new fragments
            if(window.initGraph) initGraph();
        } else {
            throw new Error(data.message);
        }
    } catch (e) {
        logToConsole("research", `Ingestion Error: ${e.message}`, "log-error");
        document.querySelector('.dropzone-content').style.display = 'block';
        document.getElementById('dropzone-status').style.display = 'none';
        alert("Knowledge drop failed. Check logs.");
    }
}

// Simple Markdown to HTML formatter for the UI
function formatMarkdown(text) {
    if (!text) return "";
    let html = text;
    // Handle code blocks
    html = html.replace(/```python([\s\S]*?)```/g, '<pre class="code-block"><code>$1</code></pre>');
    html = html.replace(/```json([\s\S]*?)```/g, '<pre class="code-block"><code>$1</code></pre>');
    html = html.replace(/```([\s\S]*?)```/g, '<pre class="code-block"><code>$1</code></pre>');
    
    // Handle Headers
    html = html.replace(/^# (.*$)/gim, '<h1>$1</h1>');
    html = html.replace(/^## (.*$)/gim, '<h2>$1</h2>');
    html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
    
    // Handle Bold/Italic
    html = html.replace(/\*\*(.*)\*\*/gim, '<strong>$1</strong>');
    html = html.replace(/\*(.*)\*/gim, '<em>$1</em>');
    
    // Handle List items
    html = html.replace(/^\- (.*$)/gim, '<li>$1</li>');
    
    // Line breaks
    html = html.replace(/\n/g, '<br>');
    
    return html;
}

/**
 * Phase 5: PDF Export logic using html2pdf.js
 */
function exportToPDF() {
    const element = document.getElementById('view-reports');
    const opt = {
        margin:       0.5,
        filename:     'aira_research_synthesis.pdf',
        image:        { type: 'jpeg', quality: 0.98 },
        html2canvas:  { scale: 2, backgroundColor: '#0f0f12', useCORS: true },
        jsPDF:        { unit: 'in', format: 'letter', orientation: 'portrait' }
    };

    logToConsole("reports", "Generating high-fidelity PDF export...", "log-step");
    html2pdf().set(opt).from(element).save().then(() => {
        logToConsole("reports", "PDF export successful.", "log-success");
    });
}
async function syncLeaderboard() {
    const body = document.getElementById('leaderboard-body');
    if(!body) return;

    try {
        const res = await fetch(`${API_BASE}/analytics/leaderboard`);
        const data = await res.json();
        
        if (data.status === "success" && data.leaderboard) {
            body.innerHTML = '';
            if (data.leaderboard.length === 0) {
                body.innerHTML = '<tr><td colspan="8" style="text-align:center; padding: 40px; color: var(--text-muted);">No papers analyzed yet. Start a research session to populate the matrix.</td></tr>';
                return;
            }

            data.leaderboard.forEach((item, index) => {
                const rank = index + 1;
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td><input type="checkbox" class="hitl-checkbox"></td>
                    <td class="rank-num">${rank}</td>
                    <td class="title-cell">
                        ${item.title}
                        <span class="source-info">${item.source_title || "Unknown Source"} (${item.year || "2026"})</span>
                    </td>
                    <td class="citescore-val">${item.citescore || "0.0"}</td>
                    <td class="percentile-cell">${item.percentile || "0"}%</td>
                    <td class="citations-val">${(item.citations || 0).toLocaleString()}</td>
                    <td>${item.documents || 0}</td>
                    <td>${item.cited_pct || 0}%</td>
                `;
                body.appendChild(row);
            });
        }
    } catch (e) {
        body.innerHTML = `<tr><td colspan="8" style="text-align:center; padding: 20px; color: var(--log-error);">Matrix Sync Failed: ${e.message}</td></tr>`;
    }
}

function connectWebSockets() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const wsUrl = `${protocol}//${host}/api/ws/logs`;
    
    console.log("Connecting to Agent Matrix Telemetry at:", wsUrl);
    
    const socket = new WebSocket(wsUrl);
    
    socket.onopen = () => {
        console.log("WebSocket connected to matrix.");
        logToConsole("research", "Telemetry synchronized with Agent Matrix.", "log-success");
        logToConsole("reports", "Telemetry synchronized with Agent Matrix.", "log-success");
    };
    
    socket.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            if (data.agent && data.message) {
                const consoleType = data.type || "research";
                logToConsole(consoleType, `[${data.agent}] ${data.message}`, data.level || "log-line");
            }
        } catch (e) {
            // Raw text log
            logToConsole("research", event.data, "log-line");
        }
    };
    
    socket.onclose = () => {
        console.warn("WebSocket disconnected. Retrying in 5s...");
        setTimeout(connectWebSockets, 5000);
    };
    
    socket.onerror = (err) => {
        console.error("WebSocket error:", err);
    };
}
