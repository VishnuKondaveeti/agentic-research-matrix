/**
 * ============================================================
 * A.I.R.A - Frontend Logic
 * ============================================================
 *
 * Handles:
 *  - SPA navigation
 *  - API requests
 *  - Live WebSocket agent telemetry
 *  - Research / ingestion
 *  - Synthesis
 *  - Chat
 *  - Analytics
 *  - Knowledge uploads
 *  - PDF / bibliography export
 *
 * ============================================================
 */


/* ============================================================
 * CONFIGURATION
 * ============================================================
 */
const API_BASE = "/api";

const WS_BASE =
    `${window.location.protocol === "https:" ? "wss:" : "ws:"}//` +
    `${window.location.host}/api/ws/logs`;


/* ============================================================
 * GLOBAL STATE
 * ============================================================
 */

let logSocket = null;

let reconnectTimer = null;

let currentProvider =
    localStorage.getItem("llm_provider") || "gemini";

let insightInterval = null;

let network = null;


/* ============================================================
 * DOM REFERENCES
 * ============================================================
 */

const navItems =
    document.querySelectorAll(".nav-item");

const views =
    document.querySelectorAll(".view");

const topTitle =
    document.getElementById("top-title");


/* ============================================================
 * INITIALIZATION
 * ============================================================
 */

document.addEventListener("DOMContentLoaded", () => {

    /* --------------------------------------------------------
     * Navigation
     * --------------------------------------------------------
     */

    navItems.forEach(item => {

        item.addEventListener("click", () => {

            const viewName =
                item.getAttribute("data-view");

            switchView(viewName);
        });
    });


    /* --------------------------------------------------------
     * Research
     * --------------------------------------------------------
     */

    const btnSearch =
        document.getElementById("btn-search-init");

    if (btnSearch) {

        btnSearch.addEventListener(
            "click",
            startResearchCascade
        );
    }


    const btnApprove =
        document.getElementById("btn-hitl-approve");

    if (btnApprove) {

        btnApprove.addEventListener(
            "click",
            approveHitlProcessor
        );
    }


    const btnCancel =
        document.getElementById("btn-hitl-cancel");

    if (btnCancel) {

        btnCancel.addEventListener(
            "click",
            () => {

                const panel =
                    document.getElementById("hitl-panel");

                if (panel) {
                    panel.style.display = "none";
                }

                logToConsole(
                    "research",
                    "Process aborted by human authorization.",
                    "log-error"
                );
            }
        );
    }


    /* --------------------------------------------------------
     * Chat
     * --------------------------------------------------------
     */

    const chatSend =
        document.getElementById("btn-chat-send");

    if (chatSend) {

        chatSend.addEventListener(
            "click",
            sendChatMessage
        );
    }


    const chatInput =
        document.getElementById("chat-input");

    if (chatInput) {

        chatInput.addEventListener(
            "keypress",
            event => {

                if (event.key === "Enter") {
                    sendChatMessage();
                }
            }
        );
    }


    /* --------------------------------------------------------
     * Analytics Tabs
     * --------------------------------------------------------
     */

    document
        .querySelectorAll(".tab-btn")
        .forEach(btn => {

            btn.addEventListener(
                "click",
                () => {

                    switchTab(
                        btn.getAttribute("data-target")
                    );
                }
            );
        });


    /* --------------------------------------------------------
     * Synthesis
     * --------------------------------------------------------
     */

    const btnGenerateReport =
        document.getElementById(
            "btn-generate-report"
        );

    if (btnGenerateReport) {

        btnGenerateReport.addEventListener(
            "click",
            generateSynthesisReport
        );
    }


    /* --------------------------------------------------------
     * Report K slider
     * --------------------------------------------------------
     */

    const reportK =
        document.getElementById("report-k");

    const reportKVal =
        document.getElementById("report-k-val");

    if (reportK && reportKVal) {

        reportK.addEventListener(
            "input",
            () => {

                reportKVal.textContent =
                    `${reportK.value} nodes`;
            }
        );
    }


    /* --------------------------------------------------------
     * LLM Provider
     * --------------------------------------------------------
     */

    const providerSelect =
        document.getElementById("provider-select");

    const modelBadge =
        document.getElementById("active-model-badge");


    if (providerSelect) {

        providerSelect.value =
            currentProvider;


        if (modelBadge) {

            modelBadge.textContent =
                capitalize(currentProvider);
        }


        providerSelect.addEventListener(
            "change",
            event => {

                currentProvider =
                    event.target.value;

                localStorage.setItem(
                    "llm_provider",
                    currentProvider
                );


                if (modelBadge) {

                    modelBadge.textContent =
                        capitalize(currentProvider);

                    modelBadge.style.animation =
                        "none";

                    modelBadge.offsetHeight;

                    modelBadge.style.animation =
                        "pulse 1s linear";
                }


                logToConsole(
                    "research",
                    `LLM Provider switched to: ${currentProvider}`,
                    "log-step"
                );
            }
        );
    }


    /* --------------------------------------------------------
     * Initial data
     * --------------------------------------------------------
     */

    fetchStats();


    /* --------------------------------------------------------
     * WebSocket telemetry
     * --------------------------------------------------------
     */

    connectWebSockets();


    /* --------------------------------------------------------
     * Initial overlay
     * --------------------------------------------------------
     */

    const dismissOverlay = () => {
        const overlay = document.getElementById("init-overlay");
        if (overlay) {
            overlay.classList.add("fade-out");
            setTimeout(() => {
                overlay.style.display = "none";
            }, 600);
        }
    };

    setTimeout(dismissOverlay, 800);
    window.addEventListener("load", () => setTimeout(dismissOverlay, 500));
});


/* ============================================================
 * UTILITY
 * ============================================================
 */

function capitalize(value) {

    if (!value) {
        return "";
    }

    return (
        value.charAt(0).toUpperCase() +
        value.slice(1)
    );
}


/* ============================================================
 * WEBSOCKET - LIVE AGENT TELEMETRY
 * ============================================================
 */

function connectWebSockets() {

    /* --------------------------------------------------------
     * Prevent duplicate connections
     * --------------------------------------------------------
     */

    if (
        logSocket &&
        (
            logSocket.readyState === WebSocket.OPEN ||
            logSocket.readyState === WebSocket.CONNECTING
        )
    ) {

        console.log(
            "[Telemetry] WebSocket already connected."
        );

        return;
    }


    /* --------------------------------------------------------
     * Clear previous reconnect timer
     * --------------------------------------------------------
     */

    if (reconnectTimer) {

        clearTimeout(
            reconnectTimer
        );

        reconnectTimer = null;
    }


    console.log(
        "[Telemetry] Connecting to:",
        WS_BASE
    );


    try {

        logSocket =
            new WebSocket(WS_BASE);

    } catch (error) {

        console.error(
            "[Telemetry] Failed to create WebSocket:",
            error
        );

        scheduleWebSocketReconnect();

        return;
    }


    /* --------------------------------------------------------
     * OPEN
     * --------------------------------------------------------
     */

    logSocket.onopen = () => {

        console.log(
            "[Telemetry] WebSocket connected."
        );


        logToConsole(
            "research",
            "Live agent telemetry connected.",
            "log-success"
        );


        logToConsole(
            "reports",
            "Live agent telemetry connected.",
            "log-success"
        );
    };


    /* --------------------------------------------------------
     * MESSAGE
     * --------------------------------------------------------
     */

    logSocket.onmessage = event => {

        try {

            const data =
                JSON.parse(event.data);


            console.log(
                "[Telemetry Event]",
                data
            );


            /* =================================================
             * AGENT LOG EVENT
             *
             * Example:
             *
             * {
             *   "type": "log",
             *   "agent": "ResearchAgent",
             *   "message": "Searching papers...",
             *   "level": "info",
             *   "channel": "research"
             * }
             * =================================================
             */

            if (
                data.agent &&
                data.message
            ) {

                let consoleType =
                    data.channel ||
                    "research";


                /* ---------------------------------------------
                 * Valid UI consoles
                 * ---------------------------------------------
                 */

                if (
                    consoleType !== "research" &&
                    consoleType !== "reports"
                ) {

                    consoleType =
                        "research";
                }


                /* ---------------------------------------------
                 * Determine CSS class
                 * ---------------------------------------------
                 */

                let levelClass =
                    "log-line";


                const level =
                    String(
                        data.level || "info"
                    ).toLowerCase();


                if (
                    level === "error" ||
                    level === "critical"
                ) {

                    levelClass =
                        "log-error";

                } else if (
                    level === "warning" ||
                    level === "warn"
                ) {

                    levelClass =
                        "log-warning";

                } else if (
                    level === "success"
                ) {

                    levelClass =
                        "log-success";

                } else if (
                    level === "step"
                ) {

                    levelClass =
                        "log-step";

                } else if (
                    data.agent ===
                    "OrchestratorAgent"
                ) {

                    levelClass =
                        "log-step";
                }


                /* ---------------------------------------------
                 * Build message
                 * ---------------------------------------------
                 */

                let message =
                    `[${data.agent}] ${data.message}`;


                /* ---------------------------------------------
                 * Optional progress
                 * ---------------------------------------------
                 */

                if (
                    typeof data.progress ===
                    "number"
                ) {

                    const progress =
                        Math.max(
                            0,
                            Math.min(
                                100,
                                data.progress
                            )
                        );


                    message +=
                        ` [${progress}%]`;
                }


                /* ---------------------------------------------
                 * Optional stage
                 * ---------------------------------------------
                 */

                if (
                    data.stage &&
                    !message.includes(
                        `[${data.stage}]`
                    )
                ) {

                    message =
                        `[${data.stage}] ${message}`;
                }


                logToConsole(
                    consoleType,
                    message,
                    levelClass
                );


                return;
            }


            /* =================================================
             * DEBATE EVENT
             * =================================================
             */

            if (
                data.type === "debate"
            ) {

                if (
                    data.agent &&
                    data.message
                ) {

                    logToWarRoom(
                        data.agent,
                        data.message
                    );
                }

                return;
            }


            /* =================================================
             * GENERIC MESSAGE
             * =================================================
             */

            if (data.message) {

                logToConsole(
                    data.channel ||
                    "research",

                    data.message,

                    data.level === "error"
                        ? "log-error"
                        : "log-line"
                );

                return;
            }


            console.log(
                "[Telemetry] Unknown event:",
                data
            );

        } catch (error) {

            console.error(
                "[Telemetry] JSON parsing failed:",
                error
            );


            /* Raw text fallback */

            logToConsole(
                "research",
                event.data,
                "log-line"
            );
        }
    };


    /* --------------------------------------------------------
     * CLOSE
     * --------------------------------------------------------
     */

    logSocket.onclose = event => {

        console.warn(
            `[Telemetry] WebSocket disconnected. ` +
            `Code=${event.code}. Retrying in 5 seconds...`
        );


        logSocket = null;


        scheduleWebSocketReconnect();
    };


    /* --------------------------------------------------------
     * ERROR
     * --------------------------------------------------------
     */

    logSocket.onerror = error => {

        console.error(
            "[Telemetry] WebSocket error:",
            error
        );
    };
}


/* ============================================================
 * WEBSOCKET RECONNECT
 * ============================================================
 */

function scheduleWebSocketReconnect() {

    if (reconnectTimer) {
        return;
    }


    reconnectTimer =
        setTimeout(
            () => {

                reconnectTimer = null;

                connectWebSockets();

            },
            5000
        );
}


/* ============================================================
 * NAVIGATION
 * ============================================================
 */

function switchView(viewName) {

    views.forEach(
        view =>
            view.classList.remove(
                "active"
            )
    );


    navItems.forEach(
        item =>
            item.classList.remove(
                "active"
            )
    );


    const activeView =
        document.getElementById(
            `view-${viewName}`
        );


    const activeNav =
        document.querySelector(
            `.nav-item[data-view="${viewName}"]`
        );


    if (
        activeView &&
        activeNav
    ) {

        activeView.classList.add(
            "active"
        );

        activeNav.classList.add(
            "active"
        );


        const titleElement =
            activeNav.querySelector("span");


        if (
            titleElement &&
            topTitle
        ) {

            topTitle.innerHTML =
                `Research / <span>${titleElement.textContent}</span>`;
        }


        /* ---------------------------------------------
         * Lazy analytics loading
         * ---------------------------------------------
         */

        if (
            viewName === "analytics"
        ) {

            switchTab(
                "tab-graph"
            );
        }
    }
}


/* ============================================================
 * TABS
 * ============================================================
 */

function switchTab(tabId) {

    document
        .querySelectorAll(".tab-btn")
        .forEach(
            button =>
                button.classList.remove(
                    "active"
                )
        );


    document
        .querySelectorAll(".tab-content")
        .forEach(
            content =>
                content.classList.remove(
                    "active"
                )
        );


    const button =
        document.querySelector(
            `.tab-btn[data-target="${tabId}"]`
        );


    const content =
        document.getElementById(
            tabId
        );


    if (
        button &&
        content
    ) {

        button.classList.add(
            "active"
        );

        content.classList.add(
            "active"
        );


        if (
            tabId === "tab-graph"
        ) {

            initGraph();
        }


        if (
            tabId === "tab-3d"
        ) {

            initVectorSpace();
        }


        if (
            tabId === "tab-trends"
        ) {

            initTrends();
        }


        if (
            tabId === "tab-leaderboard"
        ) {

            syncLeaderboard();
        }
    }
}


/* ============================================================
 * API - HEALTH
 * ============================================================
 */

async function fetchStats() {

    try {

        const res =
            await fetch(
                `${API_BASE}/health`
            );


        const data =
            await res.json();


        const element =
            document.querySelector(
                "#sidebar-db-stats .highlight"
            );


        if (element) {

            element.textContent =
                data.vector_db_documents ||
                "0";
        }

    } catch (error) {

        console.warn(
            "Backend not running yet:",
            error
        );
    }
}


/* ============================================================
 * CONSOLE LOGGING
 * ============================================================
 */

function logToConsole(
    consoleType,
    msg,
    typeClass = "log-line"
) {

    const out =
        document.getElementById(
            `console-output-${consoleType}`
        );


    /* --------------------------------------------------------
     * Prevent UI crash
     * --------------------------------------------------------
     */

    if (!out) {

        console.warn(
            `[Console] Missing console '${consoleType}':`,
            msg
        );

        return;
    }


    const div =
        document.createElement(
            "div"
        );


    div.className =
        typeClass;


    /* --------------------------------------------------------
     * Timestamp
     * --------------------------------------------------------
     */

    const timestamp =
        document.createElement(
            "span"
        );


    timestamp.style.color =
        "#5c6fff";


    timestamp.textContent =
        `[${new Date().toLocaleTimeString()}]`;


    div.appendChild(
        timestamp
    );


    div.appendChild(
        document.createTextNode(
            ` ${msg}`
        )
    );


    out.appendChild(
        div
    );


    /* --------------------------------------------------------
     * Auto-scroll
     * --------------------------------------------------------
     */

    out.scrollTop =
        out.scrollHeight;
}


/* ============================================================
 * WAR ROOM
 * ============================================================
 */

function logToWarRoom(
    agentId,
    content
) {

    const container =
        document.getElementById(
            "war-room-container"
        );


    if (!container) {
        return;
    }


    container.style.display =
        "block";


    const log =
        document.getElementById(
            "war-debate-log"
        );


    if (!log) {
        return;
    }


    const div =
        document.createElement(
            "div"
        );


    div.className =
        `debate-message ${String(
            agentId
        ).toLowerCase()}`;


    const author =
        document.createElement(
            "span"
        );


    author.className =
        "author";


    author.textContent =
        `${capitalize(agentId)}:`;


    const contentElement =
        document.createElement(
            "span"
        );


    contentElement.className =
        "content";


    contentElement.textContent =
        content;


    div.appendChild(
        author
    );


    div.appendChild(
        document.createTextNode(" ")
    );


    div.appendChild(
        contentElement
    );


    log.appendChild(
        div
    );


    log.scrollTop =
        log.scrollHeight;


    /* --------------------------------------------------------
     * Highlight speaking agent
     * --------------------------------------------------------
     */

    document
        .querySelectorAll(".war-agent-box")
        .forEach(
            box =>
                box.classList.remove(
                    "speaking"
                )
        );


    const speakerBox =
        document.getElementById(
            `war-agent-${String(
                agentId
            ).toLowerCase()}`
        );


    if (speakerBox) {

        speakerBox.classList.add(
            "speaking"
        );


        const thought =
            speakerBox.querySelector(
                ".war-thought"
            );


        if (thought) {

            thought.innerText =
                content;
        }
    }
}


/* ============================================================
 * INSIGHT STREAM
 * ============================================================
 */

function startInsightStream() {

    const streamer =
        document.getElementById(
            "insight-streamer"
        );


    if (!streamer) {
        return;
    }


    const facts = [

        "Analyzing multidimensional vector manifolds...",

        "Traversing knowledge topology...",

        "Cross-referencing global paper indices...",

        "Optimizing RAG retrieval pathways...",

        "Detecting emerging research patterns...",

        "Mapping researcher influence via citation graphs...",

        "Preparing evidence for synthesis...",

        "Running autonomous research cascade..."

    ];


    let index = 0;


    streamer.innerHTML =
        `<div class="fact-text">${facts[0]}</div>`;


    if (insightInterval) {

        clearInterval(
            insightInterval
        );
    }


    insightInterval =
        setInterval(
            () => {

                index++;

                streamer.innerHTML =
                    `<div class="fact-text">${
                        facts[
                            index %
                            facts.length
                        ]
                    }</div>`;

            },
            3000
        );
}


function stopInsightStream() {

    if (insightInterval) {

        clearInterval(
            insightInterval
        );

        insightInterval =
            null;
    }
}


/* ============================================================
 * RESEARCH
 * ============================================================
 */

async function startResearchCascade() {

    const queryInput =
        document.getElementById(
            "research-query"
        );


    const query =
        queryInput
            ? queryInput.value.trim()
            : "";


    if (!query) {

        alert(
            "Enter a target topic"
        );

        return;
    }


    const loader =
        document.getElementById(
            "neural-pulse-loader"
        );


    const hitl =
        document.getElementById(
            "hitl-panel"
        );


    const consoleElement =
        document.getElementById(
            "agent-console-research"
        );


    if (loader) {
        loader.style.display = "flex";
    }


    if (hitl) {
        hitl.style.display = "none";
    }


    if (consoleElement) {
        consoleElement.style.display = "block";
    }


    startInsightStream();


    logToConsole(
        "research",
        `Starting research search: ${query}`,
        "log-step"
    );


    try {

        const maxPapersElement =
            document.getElementById(
                "research-max"
            );


        const complexityElement =
            document.getElementById(
                "research-complexity"
            );


        const res =
            await fetch(
                `${API_BASE}/search`,
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({

                        query:

                            query,

                        max_papers:

                            maxPapersElement
                                ? parseInt(
                                    maxPapersElement.value
                                )
                                : 5,

                        complexity:

                            complexityElement
                                ? complexityElement.value
                                : "medium",

                        sources: [
                            "arxiv"
                        ],

                        download: false,

                        process: false,

                        llm_provider:
                            currentProvider
                    })
                }
            );


        const data =
            await res.json();


        stopInsightStream();


        if (loader) {
            loader.style.display = "none";
        }


        if (
            data.status ===
            "success"
        ) {

            showHITLPanel(
                query,
                data.papers || []
            );


            if (consoleElement) {

                consoleElement.style.display =
                    "block";
            }

        } else {

            logToConsole(
                "research",
                `Search failed: ${
                    data.message ||
                    "Unknown error"
                }`,
                "log-error"
            );
        }

    } catch (error) {

        stopInsightStream();


        if (loader) {
            loader.style.display = "none";
        }


        logToConsole(
            "research",
            `Network error: ${error.message}`,
            "log-error"
        );
    }
}


/* ============================================================
 * HITL
 * ============================================================
 */

function showHITLPanel(
    query,
    papers
) {

    const panel =
        document.getElementById(
            "hitl-panel"
        );


    const list =
        document.getElementById(
            "hitl-paper-list"
        );


    const count =
        document.getElementById(
            "hitl-count"
        );


    if (!panel || !list) {
        return;
    }


    panel.style.display =
        "block";


    list.innerHTML =
        "";


    if (count) {

        count.textContent =
            papers.length;
    }


    papers.forEach(
        paper => {

            const item =
                document.createElement(
                    "div"
                );


            item.style.padding =
                "10px";


            item.style.borderBottom =
                "1px solid rgba(255,255,255,0.05)";


            item.style.display =
                "flex";


            item.style.alignItems =
                "center";


            item.style.gap =
                "12px";


            const score =
                paper.influence_score ||
                0;


            const scoreColor =
                score > 80
                    ? "#4ade80"
                    : score > 60
                        ? "#fbbf24"
                        : "#9ca3af";


            item.innerHTML = `
                <label
                    class="cb-container"
                    style="flex-shrink:0;"
                >
                    <input
                        type="checkbox"
                        checked
                        data-pid="${escapeHtml(
                            paper.id ||
                            paper.title ||
                            ""
                        )}"
                        data-score="${score}"
                        data-title="${escapeHtml(
                            paper.title ||
                            ""
                        )}"
                    >
                    <span class="checkmark"></span>
                </label>

                <div style="flex-grow:1;">
                    <div
                        style="
                            font-weight:600;
                            font-size:0.95rem;
                        "
                    >
                        ${escapeHtml(
                            paper.title ||
                            "Untitled"
                        )}
                    </div>

                    <div
                        style="
                            font-size:0.8rem;
                            color:var(--text-muted);
                        "
                    >
                        <span class="source-pill">
                            ${escapeHtml(
                                paper.source ||
                                "Unknown"
                            )}
                        </span>

                        ${escapeHtml(
                            paper.published ||
                            ""
                        )}
                    </div>
                </div>

                <div
                    class="influence-score-pill"
                    style="
                        border-color:${scoreColor};
                        color:${scoreColor};
                    "
                >
                    <i class="fa-solid fa-fire"></i>
                    ${score || 65}
                </div>
            `;


            list.appendChild(
                item
            );
        }
    );
}


/* ============================================================
 * APPROVE HITL
 * ============================================================
 */

async function approveHitlProcessor() {

    const panel =
        document.getElementById(
            "hitl-panel"
        );


    if (panel) {

        panel.style.display =
            "none";
    }


    const selected =
        Array.from(
            document.querySelectorAll(
                "#hitl-paper-list input:checked"
            )
        ).map(
            element =>
                element.getAttribute(
                    "data-pid"
                )
        );


    logToConsole(
        "research",
        "Human authorization received. Resuming Subspace Cascade...",
        "log-step"
    );


    try {

        const queryElement =
            document.getElementById(
                "research-query"
            );


        const styleElement =
            document.getElementById(
                "research-style"
            );


        const res =
            await fetch(
                `${API_BASE}/ingest`,
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({

                        query:
                            queryElement
                                ? queryElement.value
                                : "",

                        paper_ids:
                            selected,

                        style:
                            styleElement
                                ? styleElement.value
                                : "Standard",

                        llm_provider:
                            currentProvider
                    })
                }
            );


        const data =
            await res.json();


        if (
            data.status === "success" ||
            data.stages
        ) {

            logToConsole(
                "research",
                "Cascade completed successfully.",
                "log-success"
            );


            fetchStats();


            if (data.stages) {

                const stages =
                    data.stages;


                const blueprint =
                    stages.blueprint
                        ?.blueprint ||
                    "No blueprint generated.";


                const debate =
                    stages.debate
                        ?.debate ||
                    "No debate generated.";


                const hypotheses =
                    stages.hypotheses
                        ?.hypotheses ||
                    "No hypotheses generated.";


                setMarkdown(
                    "blueprint-content",
                    blueprint
                );


                setMarkdown(
                    "debate-content",
                    debate
                );


                setMarkdown(
                    "hypothesis-content",
                    hypotheses
                );


                const elite =
                    document.getElementById(
                        "elite-insights"
                    );


                if (elite) {

                    elite.style.display =
                        "block";
                }


                if (
                    stages.analysis
                ) {

                    const reportContent =
                        document.getElementById(
                            "report-content"
                        );


                    const emptyState =
                        document.getElementById(
                            "report-empty"
                        );


                    const analysisText =
                        stages.analysis.analysis ||
                        stages.analysis.summary ||
                        "";


                    if (reportContent) {

                        reportContent.innerHTML =
                            formatMarkdown(
                                analysisText
                            );

                        reportContent.style.display =
                            "block";
                    }


                    if (emptyState) {

                        emptyState.style.display =
                            "none";
                    }


                    const actions =
                        document.getElementById(
                            "report-actions"
                        );


                    if (actions) {

                        actions.style.display =
                            "flex";
                    }
                }


                setTimeout(
                    () => {

                        switchView(
                            "reports"
                        );


                        logToConsole(
                            "reports",
                            "Elite insights synthesized and ready for review.",
                            "log-success"
                        );

                    },
                    1000
                );
            }


            if (
                data.stages?.analysis
            ) {

                const gapContainer =
                    document.getElementById(
                        "gap-alert-container"
                    );


                if (gapContainer) {

                    gapContainer.style.display =
                        "block";
                }
            }

        } else {

            logToConsole(
                "research",
                `Ingestion failed: ${
                    data.message ||
                    "Unknown error"
                }`,
                "log-error"
            );
        }

    } catch (error) {

        logToConsole(
            "research",
            `Network error: ${error.message}`,
            "log-error"
        );
    }
}


/* ============================================================
 * SYNTHESIS
 * ============================================================
 */

async function generateSynthesisReport() {

    const topicElement =
        document.getElementById(
            "report-topic"
        );


    const complexityElement =
        document.getElementById(
            "report-complexity"
        );


    const styleElement =
        document.getElementById(
            "report-type"
        );


    const topic =
        topicElement
            ? topicElement.value.trim()
            : "";


    const complexity =
        complexityElement
            ? complexityElement.value
            : "medium";


    const style =
        styleElement
            ? styleElement.value
            : "Standard";


    if (!topic) {

        alert(
            "Enter a synthesis topic"
        );

        return;
    }


    const reporterOutput =
        document.getElementById(
            "report-content"
        );


    const emptyState =
        document.getElementById(
            "report-empty"
        );


    const agentConsole =
        document.getElementById(
            "agent-console-reports"
        );


    if (reporterOutput) {

        reporterOutput.style.display =
            "none";
    }


    if (emptyState) {

        emptyState.style.display =
            "none";
    }


    if (agentConsole) {

        agentConsole.style.display =
            "block";
    }


    logToConsole(
        "reports",
        `Initializing Synthesis Cascade for: ${topic}`,
        "log-step"
    );


    try {

        const reportKElement =
            document.getElementById(
                "report-k"
            );


        const res =
            await fetch(
                `${API_BASE}/report`,
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({

                        topic:
                            topic,

                        complexity:
                            complexity,

                        style:
                            style,

                        top_k:
                            reportKElement
                                ? parseInt(
                                    reportKElement.value
                                )
                                : 5,

                        llm_provider:
                            currentProvider
                    })
                }
            );


        const data =
            await res.json();


        if (
            data.status ===
            "success"
        ) {

            if (data.report && data.report.trim()) {
                if (reporterOutput) {
                    reporterOutput.innerHTML = formatMarkdown(data.report);
                    reporterOutput.style.display = "block";
                }

                const reportActions = document.getElementById("report-actions");
                if (reportActions) {
                    reportActions.style.display = "flex";
                }

                logToConsole("reports", "Synthesis completed successfully.", "log-success");
            } else {
                if (reporterOutput) {
                    const errMsg = data.message || "Synthesis ended without final report output (likely rate limit or quota exceeded).";
                    reporterOutput.innerHTML = `<div class="glass-panel" style="padding: 20px; border-left: 4px solid var(--log-error); margin-top: 20px;">
                        <h4 style="color: var(--log-error); margin-bottom: 8px;"><i class="fa-solid fa-triangle-exclamation"></i> Synthesis Interrupted</h4>
                        <p style="color: var(--text-muted);">${escapeHtml(errMsg)}</p>
                    </div>`;
                    reporterOutput.style.display = "block";
                }
                const reportActions = document.getElementById("report-actions");
                if (reportActions) {
                    reportActions.style.display = "none";
                }
                logToConsole("reports", `Synthesis incomplete: ${data.message || "No report generated"}`, "log-error");
            }
        }

    } catch (error) {

        logToConsole(
            "reports",
            `Network error: ${error.message}`,
            "log-error"
        );
    }
}


/* ============================================================
 * CHAT
 * ============================================================
 */

async function sendChatMessage() {

    const input =
        document.getElementById(
            "chat-input"
        );


    if (!input) {
        return;
    }


    const msg =
        input.value.trim();


    const complexityElement =
        document.getElementById(
            "chat-complexity"
        );


    const complexity =
        complexityElement
            ? complexityElement.value
            : "medium";


    if (!msg) {
        return;
    }


    appendMessage(
        "user",
        msg
    );


    input.value =
        "";


    const thoughtStream =
        document.getElementById(
            "chat-thought-stream"
        );


    if (thoughtStream) {

        thoughtStream.style.display =
            "block";
    }


    const thoughts = [

        "Analyzing intent...",

        "Querying ChromaDB Top-K = 5...",

        "Traversing retrieval context...",

        "Applying complexity constraints..."

    ];


    let index =
        0;


    const textSpan =
        thoughtStream
            ?.querySelector(
                ".thought-text"
            );


    if (textSpan) {

        textSpan.textContent =
            thoughts[0];
    }


    const interval =
        setInterval(
            () => {

                index++;


                if (
                    index <
                    thoughts.length
                ) {

                    if (textSpan) {

                        textSpan.textContent =
                            thoughts[index];
                    }
                }

            },
            700
        );


    try {

        const res =
            await fetch(
                `${API_BASE}/query`,
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({

                        query:
                            msg,

                        complexity:
                            complexity,

                        top_k:
                            5,

                        llm_provider:
                            currentProvider
                    })
                }
            );


        const data =
            await res.json();


        clearInterval(
            interval
        );


        if (thoughtStream) {

            thoughtStream.style.display =
                "none";
        }


        if (data.answer) {

            appendMessage(
                "assistant",
                data.answer
            );


            if (
                data.sources &&
                data.sources.length > 0
            ) {

                const sourceList =
                    data.sources
                        .map(
                            source =>
                                `[${source.title}]`
                        )
                        .join(", ");


                const messages =
                    document.querySelectorAll(
                        ".message.assistant"
                    );


                const lastMessage =
                    messages[
                        messages.length - 1
                    ];


                if (lastMessage) {

                    const content =
                        lastMessage.querySelector(
                            ".content"
                        );


                    if (content) {

                        const sourcesDiv =
                            document.createElement(
                                "div"
                            );


                        sourcesDiv.className =
                            "source-list-mini";


                        sourcesDiv.style.fontSize =
                            "0.75rem";


                        sourcesDiv.style.opacity =
                            "0.6";


                        sourcesDiv.style.marginTop =
                            "8px";


                        sourcesDiv.textContent =
                            `Sources: ${sourceList}`;


                        content.appendChild(
                            sourcesDiv
                        );
                    }
                }
            }

        } else {

            appendMessage(
                "assistant",
                `Error: ${
                    data.message ||
                    "Unknown error"
                }`
            );
        }

    } catch (error) {

        clearInterval(
            interval
        );


        if (thoughtStream) {

            thoughtStream.style.display =
                "none";
        }


        appendMessage(
            "assistant",
            "Network Error: Could not reach the agent cluster. Ensure FastAPI is running."
        );
    }
}


/* ============================================================
 * CHAT MESSAGE
 * ============================================================
 */

function appendMessage(
    role,
    content
) {

    const history =
        document.getElementById(
            "chat-history"
        );


    if (!history) {
        return;
    }


    const div =
        document.createElement(
            "div"
        );


    div.className =
        `message ${role}`;


    const icon =
        role === "user"
            ? '<i class="fa-solid fa-user"></i>'
            : '<i class="fa-solid fa-brain"></i>';


    const avatar =
        document.createElement(
            "div"
        );


    avatar.className =
        "avatar";


    avatar.innerHTML =
        icon;


    const contentElement =
        document.createElement(
            "div"
        );


    contentElement.className =
        "content";


    if (
        role === "assistant"
    ) {

        contentElement.innerHTML =
            formatMarkdown(
                content
            );

    } else {

        contentElement.textContent =
            content;
    }


    div.appendChild(
        avatar
    );


    div.appendChild(
        contentElement
    );


    history.appendChild(
        div
    );


    history.scrollTop =
        history.scrollHeight;
}


/* ============================================================
 * ANALYTICS - GRAPH
 * ============================================================
 */

async function initGraph() {

    const container =
        document.getElementById(
            "d3-graph-canvas"
        );


    if (
        !container ||
        !window.vis
    ) {
        return;
    }


    try {

        const res =
            await fetch(
                `${API_BASE}/analytics/graph`
            );


        const graphData =
            await res.json();


        if (
            graphData.status !==
            "success"
        ) {

            throw new Error(
                "Graph sync failed"
            );
        }


        const nodes =
            new vis.DataSet(
                graphData.graph.nodes
            );


        const edges =
            new vis.DataSet(
                graphData.graph.edges
            );


        const data = {
            nodes,
            edges
        };


        const options = {

            nodes: {

                shape: "dot",

                size: 25,

                font: {
                    size: 14,
                    color: "#f8f8f2",
                    face: "Inter"
                },

                borderWidth: 2,

                shadow: true
            },


            edges: {

                width: 2,

                color: {
                    inherit: "from"
                },

                smooth: {
                    type: "continuous"
                }
            },


            groups: {

                core: {
                    color: {
                        background: "#5c6fff",
                        border: "#3a4dfa"
                    },
                    size: 30
                },

                agent: {
                    color: {
                        background: "#ff5c8d",
                        border: "#fa3a74"
                    }
                },

                storage: {
                    color: {
                        background: "#00f0ff",
                        border: "#00c8d7"
                    }
                },

                source: {
                    color: {
                        background: "#00ffaa",
                        border: "#00d78f"
                    }
                }
            },


            physics: {

                forceAtlas2Based: {

                    gravitationalConstant:
                        -100,

                    centralGravity:
                        0.01,

                    springLength:
                        150
                },

                maxVelocity:
                    50,

                solver:
                    "forceAtlas2Based",

                timestep:
                    0.35,

                stabilization: {

                    iterations:
                        150
                }
            }
        };


        if (network) {

            network.destroy();
        }


        network =
            new vis.Network(
                container,
                data,
                options
            );


        network.on(
            "click",
            params => {

                if (
                    params.nodes.length >
                    0
                ) {

                    const nodeId =
                        params.nodes[0];


                    const node =
                        nodes.get(
                            nodeId
                        );


                    if (node) {

                        logToConsole(
                            "research",

                            `Investigating node: ${node.label} - ${node.title || ""}`,

                            "log-step"
                        );
                    }
                }
            }
        );

    } catch (error) {

        console.error(
            "Graph init failed",
            error
        );
    }
}


/* ============================================================
 * ANALYTICS - VECTOR SPACE
 * ============================================================
 */

async function initVectorSpace() {

    if (!window.Plotly) {
        return;
    }


    const container =
        document.getElementById(
            "plotly-3d-canvas"
        );


    if (!container) {
        return;
    }


    try {

        const res =
            await fetch(
                `${API_BASE}/analytics/embeddings?dimensions=3`
            );


        const vectorData =
            await res.json();


        if (
            vectorData.status !==
            "success"
        ) {

            throw new Error(
                "Vector sync failed"
            );
        }


        const data = [{

            x:
                vectorData.points.map(
                    point => point.x
                ),

            y:
                vectorData.points.map(
                    point => point.y
                ),

            z:
                vectorData.points.map(
                    point => point.z
                ),

            text:
                vectorData.points.map(
                    point => point.title
                ),

            mode:
                "markers",

            marker: {

                size: 4,

                color:
                    "#5c6fff",

                opacity:
                    0.8,

                line: {

                    width:
                        0.5,

                    color:
                        "rgba(255,255,255,0.1)"
                }
            },

            type:
                "scatter3d"

        }];


        const layout = {

            paper_bgcolor:
                "rgba(0,0,0,0)",

            plot_bgcolor:
                "rgba(0,0,0,0)",

            margin: {
                l: 0,
                r: 0,
                b: 0,
                t: 0
            },

            scene: {

                xaxis: {
                    showgrid: false,
                    zeroline: false,
                    showbackground: false,
                    visible: false
                },

                yaxis: {
                    showgrid: false,
                    zeroline: false,
                    showbackground: false,
                    visible: false
                },

                zaxis: {
                    showgrid: false,
                    zeroline: false,
                    showbackground: false,
                    visible: false
                },

                camera: {

                    eye: {
                        x: 1.5,
                        y: 1.5,
                        z: 1.5
                    }
                }
            },

            font: {
                color:
                    "#6e6e73"
            }
        };


        Plotly.newPlot(
            "plotly-3d-canvas",
            data,
            layout,
            {
                displayModeBar:
                    false,

                responsive:
                    true
            }
        );

    } catch (error) {

        console.error(
            "Vector Space init failed",
            error
        );
    }
}


/* ============================================================
 * ANALYTICS - TRENDS
 * ============================================================
 */

async function initTrends() {

    const timelineContainer =
        document.getElementById(
            "timeline-chart"
        );


    if (!timelineContainer) {
        return;
    }


    try {

        const trace = {

            x: [
                "2022",
                "2023",
                "2024",
                "2025",
                "2026"
            ],

            y: [
                12,
                18,
                25,
                42,
                65
            ],

            type:
                "bar",

            marker: {
                color:
                    "#5c6fff"
            }
        };


        const layout = {

            paper_bgcolor:
                "rgba(0,0,0,0)",

            plot_bgcolor:
                "rgba(0,0,0,0)",

            font: {

                color:
                    "#6e6e73",

                family:
                    "Inter"
            },

            margin: {
                l: 40,
                r: 20,
                t: 40,
                b: 40
            },

            xaxis: {
                gridcolor:
                    "rgba(0,0,0,0.05)"
            },

            yaxis: {
                gridcolor:
                    "rgba(0,0,0,0.05)"
            },

            showlegend:
                false
        };


        Plotly.newPlot(
            "timeline-chart",
            [trace],
            layout,
            {
                responsive:
                    true,

                displayModeBar:
                    false
            }
        );


        const researcher =
            document.getElementById(
                "global-top-researcher"
            );


        const trending =
            document.getElementById(
                "global-trending-topic"
            );


        const discussed =
            document.getElementById(
                "global-most-discussed"
            );


        if (researcher) {
            researcher.innerText =
                "Yann LeCun";
        }


        if (trending) {
            trending.innerText =
                "World Models + RAG";
        }


        if (discussed) {
            discussed.innerText =
                "JEPA Architecture";
        }

    } catch (error) {

        console.error(
            "Trends init failed",
            error
        );
    }
}


/* ============================================================
 * GLOBAL TRENDS SEARCH
 * ============================================================
 */

async function searchGlobalTrends() {

    const input =
        document.getElementById(
            "trends-search-input"
        );


    const query =
        input
            ? input.value.trim()
            : "";


    if (!query) {

        alert(
            "Enter a name or topic"
        );

        return;
    }


    logToConsole(
        "research",
        `Querying Global Trends for: ${query}...`,
        "log-step"
    );


    try {

        const res =
            await fetch(
                `${API_BASE}/trends/${encodeURIComponent(query)}`
            );


        const data =
            await res.json();


        if (data.error) {

            logToConsole(
                "research",
                `Trends Error: ${data.error}`,
                "log-warning"
            );

            return;
        }


        if (
            data.top_authors &&
            data.top_authors.length >
            0
        ) {

            const element =
                document.getElementById(
                    "global-top-researcher"
                );


            if (element) {

                element.textContent =
                    data.top_authors[0].name;
            }

        } else {

            const element =
                document.getElementById(
                    "global-top-researcher"
                );


            if (element) {

                element.textContent =
                    query;
            }
        }


        if (
            data.top_keywords &&
            data.top_keywords.length >
            0
        ) {

            const element =
                document.getElementById(
                    "global-trending-topic"
                );


            if (element) {

                element.textContent =
                    data.top_keywords[0].keyword;
            }
        }


        if (
            data.papers &&
            data.papers.length >
            0
        ) {

            const validPaper =
                data.papers.find(
                    paper =>
                        paper.title &&
                        paper.title !==
                        "Untitled"
                );


            if (validPaper) {

                const element =
                    document.getElementById(
                        "global-most-discussed"
                    );


                if (element) {

                    element.textContent =
                        validPaper.title;
                }
            }
        }


        if (
            data.timeline &&
            data.timeline.length >
            0
        ) {

            const x =
                data.timeline.map(
                    item =>
                        item.period
                );


            const y =
                data.timeline.map(
                    item =>
                        item.count
                );


            const trace = {

                x,
                y,

                type:
                    "bar",

                marker: {
                    color:
                        "#5c6fff"
                }
            };


            const layout = {

                paper_bgcolor:
                    "rgba(0,0,0,0)",

                plot_bgcolor:
                    "rgba(0,0,0,0)",

                font: {

                    color:
                        "#6e6e73",

                    family:
                        "Inter"
                },

                margin: {
                    l: 40,
                    r: 20,
                    t: 40,
                    b: 40
                },

                xaxis: {
                    gridcolor:
                        "rgba(0,0,0,0.05)"
                },

                yaxis: {
                    gridcolor:
                        "rgba(0,0,0,0.05)"
                },

                showlegend:
                    false
            };


            Plotly.newPlot(
                "timeline-chart",
                [trace],
                layout,
                {
                    responsive:
                        true,

                    displayModeBar:
                        false
                }
            );
        }


        if (
            window.initGraph
        ) {

            initGraph();
        }


        logToConsole(
            "research",
            `Analytics synchronized for: ${query}`,
            "log-success"
        );

    } catch (error) {

        logToConsole(
            "research",
            `Trends Network Error: ${error.message}`,
            "log-error"
        );
    }
}


/* ============================================================
 * PAPER SELECTION
 * ============================================================
 */

function selectAllPapers(
    checked
) {

    document
        .querySelectorAll(
            "#hitl-paper-list input"
        )
        .forEach(
            checkbox =>
                checkbox.checked =
                    checked
        );
}


function filterHighImpact() {

    document
        .querySelectorAll(
            "#hitl-paper-list input"
        )
        .forEach(
            checkbox => {

                const score =
                    parseInt(
                        checkbox.getAttribute(
                            "data-score"
                        )
                    ) || 0;


                checkbox.checked =
                    score >= 80;
            }
        );
}


/* ============================================================
 * BIBLIOGRAPHY
 * ============================================================
 */

function exportBibliography() {

    const checkboxes =
        Array.from(
            document.querySelectorAll(
                "#hitl-paper-list input:checked"
            )
        );


    const papers =
        checkboxes.map(
            checkbox =>
                checkbox.getAttribute(
                    "data-title"
                )
        );


    if (
        papers.length ===
        0
    ) {

        alert(
            "Search for papers first to generate a bibliography."
        );

        return;
    }


    const bibtex =
        papers
            .map(
                (title, index) => {

                    return (
                        `@article{paper${index},\n` +
                        `  title={${title}},\n` +
                        `  author={AIRA Matrix Agents},\n` +
                        `  year={2026}\n` +
                        `}`
                    );
                }
            )
            .join("\n\n");


    const blob =
        new Blob(
            [bibtex],
            {
                type:
                    "text/plain"
            }
        );


    const url =
        URL.createObjectURL(
            blob
        );


    const anchor =
        document.createElement(
            "a"
        );


    anchor.href =
        url;


    anchor.download =
        "aira_bibliography.bib";


    anchor.click();


    URL.revokeObjectURL(
        url
    );


    logToConsole(
        "reports",
        "Bibliography exported successfully.",
        "log-success"
    );
}


/* ============================================================
 * KNOWLEDGE DROPZONE
 * ============================================================
 */

const dropzone =
    document.getElementById(
        "knowledge-dropzone"
    );


const dropzoneInput =
    document.getElementById(
        "dropzone-input"
    );


if (dropzone) {

    dropzone.addEventListener(
        "dragover",
        event => {

            event.preventDefault();

            dropzone.classList.add(
                "drag-over"
            );
        }
    );


    dropzone.addEventListener(
        "dragleave",
        () => {

            dropzone.classList.remove(
                "drag-over"
            );
        }
    );


    dropzone.addEventListener(
        "drop",
        event => {

            event.preventDefault();

            dropzone.classList.remove(
                "drag-over"
            );


            const files =
                event.dataTransfer.files;


            if (
                files.length >
                0
            ) {

                handleFileUpload(
                    files
                );
            }
        }
    );


    if (dropzoneInput) {

        dropzoneInput.addEventListener(
            "change",
            event => {

                if (
                    event.target.files.length >
                    0
                ) {

                    handleFileUpload(
                        event.target.files
                    );
                }
            }
        );
    }
}


/* ============================================================
 * FILE UPLOAD
 * ============================================================
 */

async function handleFileUpload(
    files
) {

    const formData =
        new FormData();


    for (
        let i = 0;
        i < files.length;
        i++
    ) {

        if (
            files[i].type !==
            "application/pdf"
        ) {

            alert(
                "Only PDFs are allowed in the elite subspace."
            );

            return;
        }


        formData.append(
            "file",
            files[i]
        );
    }


    const content =
        document.querySelector(
            ".dropzone-content"
        );


    const status =
        document.getElementById(
            "dropzone-status"
        );


    if (content) {

        content.style.display =
            "none";
    }


    if (status) {

        status.style.display =
            "flex";
    }


    logToConsole(
        "research",
        `Ingesting ${files.length} local documents...`,
        "log-step"
    );


    try {

        const res =
            await fetch(
                `${API_BASE}/upload`,
                {
                    method:
                        "POST",

                    body:
                        formData
                }
            );


        const data =
            await res.json();


        if (
            data.status ===
            "success"
        ) {

            logToConsole(
                "research",
                "Local ingestion complete. Knowledge integrated into vector space.",
                "log-success"
            );


            const statusText =
                document.getElementById(
                    "dropzone-status-text"
                );


            if (statusText) {

                statusText.innerHTML =
                    `<span style="color:#4ade80">Ingestion Successful!</span><br>${files.length} papers analyzed.`;
            }


            setTimeout(
                () => {

                    if (content) {

                        content.style.display =
                            "block";
                    }


                    if (status) {

                        status.style.display =
                            "none";
                    }


                    if (statusText) {

                        statusText.textContent =
                            "Ingesting local wisdom...";
                    }

                },
                5000
            );


            if (
                window.initGraph
            ) {

                initGraph();
            }

        } else {

            throw new Error(
                data.message ||
                "Upload failed"
            );
        }

    } catch (error) {

        logToConsole(
            "research",
            `Ingestion Error: ${error.message}`,
            "log-error"
        );


        if (content) {

            content.style.display =
                "block";
        }


        if (status) {

            status.style.display =
                "none";
        }


        alert(
            "Knowledge drop failed. Check logs."
        );
    }
}


/* ============================================================
 * WAR ROOM SIMULATION
 * ============================================================
 */

async function simulateWarRoomDebate(
    topic
) {

    const container =
        document.getElementById(
            "war-room-container"
        );


    if (!container) {
        return;
    }


    container.style.display =
        "block";


    const log =
        document.getElementById(
            "war-debate-log"
        );


    if (log) {

        log.innerHTML =
            "";
    }


    const debateSteps = [

        {
            agent:
                "critic",

            msg:
                `The methodology in the latest papers for "${topic}" seems over-reliant on synthetic benchmarks. We need more real-world validation.`
        },

        {
            agent:
                "optimist",

            msg:
                "While synthetic data is a concern, the scaling laws shown in these clusters suggest massive efficiency gains in zero-shot tasks!"
        },

        {
            agent:
                "advisor",

            msg:
                "I recommend a cross-domain verification. We should pivot our next inquiry towards the research gap in robotics integration."
        },

        {
            agent:
                "critic",

            msg:
                "Agreed. But only if we can verify the latency constraints in those robotics environments."
        }

    ];


    let step =
        0;


    logToWarRoom(
        debateSteps[
            step
        ].agent,

        debateSteps[
            step
        ].msg
    );


    step++;


    const interval =
        setInterval(
            () => {

                if (
                    step <
                    debateSteps.length
                ) {

                    logToWarRoom(
                        debateSteps[
                            step
                        ].agent,

                        debateSteps[
                            step
                        ].msg
                    );


                    step++;

                } else {

                    clearInterval(
                        interval
                    );


                    setTimeout(
                        () => {

                            document
                                .querySelectorAll(
                                    ".war-agent-box"
                                )
                                .forEach(
                                    box =>
                                        box.classList.remove(
                                            "speaking"
                                        )
                                );

                        },
                        2000
                    );
                }

            },
            4000
        );
}


/* ============================================================
 * LEADERBOARD
 * ============================================================
 */

async function syncLeaderboard() {

    const body =
        document.getElementById(
            "leaderboard-body"
        );


    if (!body) {
        return;
    }


    try {

        const res =
            await fetch(
                `${API_BASE}/analytics/leaderboard`
            );


        const data =
            await res.json();


        if (
            data.status ===
                "success" &&
            data.leaderboard
        ) {

            body.innerHTML =
                "";


            if (
                data.leaderboard.length ===
                0
            ) {

                body.innerHTML =
                    `
                    <tr>
                        <td
                            colspan="8"
                            style="
                                text-align:center;
                                padding:40px;
                                color:var(--text-muted);
                            "
                        >
                            No papers analyzed yet.
                            Start a research session
                            to populate the matrix.
                        </td>
                    </tr>
                    `;

                return;
            }


            data.leaderboard.forEach(
                (item, index) => {

                    const row =
                        document.createElement(
                            "tr"
                        );


                    const rank =
                        index + 1;


                    row.innerHTML = `
                        <td>
                            <input
                                type="checkbox"
                                class="hitl-checkbox"
                            >
                        </td>

                        <td class="rank-num">
                            ${rank}
                        </td>

                        <td class="title-cell">

                            ${escapeHtml(
                                item.title ||
                                "Untitled"
                            )}

                            <span
                                class="source-info"
                            >
                                ${escapeHtml(
                                    item.source_title ||
                                    "Unknown Source"
                                )}
                                (${item.year || "2026"})
                            </span>

                        </td>

                        <td class="citescore-val">
                            ${item.citescore || "0.0"}
                        </td>

                        <td class="percentile-cell">
                            ${item.percentile || "0"}%
                        </td>

                        <td class="citations-val">
                            ${(
                                item.citations ||
                                0
                            ).toLocaleString()}
                        </td>

                        <td>
                            ${item.documents || 0}
                        </td>

                        <td>
                            ${item.cited_pct || 0}%
                        </td>
                    `;


                    body.appendChild(
                        row
                    );
                }
            );
        }

    } catch (error) {

        body.innerHTML =
            `
            <tr>
                <td
                    colspan="8"
                    style="
                        text-align:center;
                        padding:20px;
                        color:var(--log-error);
                    "
                >
                    Matrix Sync Failed:
                    ${escapeHtml(
                        error.message
                    )}
                </td>
            </tr>
            `;
    }
}


/* ============================================================
 * MARKDOWN
 * ============================================================
 */

function formatMarkdown(
    text
) {

    if (!text) {
        return "";
    }


    let html =
        escapeHtml(
            String(text)
        );


    /* Code blocks */

    html =
        html.replace(
            /```python([\s\S]*?)```/g,
            '<pre class="code-block"><code>$1</code></pre>'
        );


    html =
        html.replace(
            /```json([\s\S]*?)```/g,
            '<pre class="code-block"><code>$1</code></pre>'
        );


    html =
        html.replace(
            /```([\s\S]*?)```/g,
            '<pre class="code-block"><code>$1</code></pre>'
        );


    /* Headers */

    html =
        html.replace(
            /^### (.*$)/gim,
            "<h3>$1</h3>"
        );


    html =
        html.replace(
            /^## (.*$)/gim,
            "<h2>$1</h2>"
        );


    html =
        html.replace(
            /^# (.*$)/gim,
            "<h1>$1</h1>"
        );


    /* Bold */

    html =
        html.replace(
            /\*\*(.*?)\*\*/g,
            "<strong>$1</strong>"
        );


    /* Italic */

    html =
        html.replace(
            /\*(.*?)\*/g,
            "<em>$1</em>"
        );


    /* Lists */

    html =
        html.replace(
            /^\- (.*$)/gim,
            "<li>$1</li>"
        );


    /* Line breaks */

    html =
        html.replace(
            /\n/g,
            "<br>"
        );


    return html;
}


/* ============================================================
 * SET MARKDOWN
 * ============================================================
 */

function setMarkdown(
    elementId,
    content
) {

    const element =
        document.getElementById(
            elementId
        );


    if (element) {

        element.innerHTML =
            formatMarkdown(
                content
            );
    }
}


/* ============================================================
 * HTML ESCAPE
 * ============================================================
 */

function escapeHtml(
    value
) {

    if (
        value === null ||
        value === undefined
    ) {

        return "";
    }


    const div =
        document.createElement(
            "div"
        );


    div.textContent =
        String(value);


    return div.innerHTML;
}


/* ============================================================
 * PDF EXPORT
 * ============================================================
 */

function exportToPDF() {

    const element =
        document.getElementById(
            "view-reports"
        );


    if (!element) {
        return;
    }


    const opt = {

        margin:
            0.5,

        filename:
            "aira_research_synthesis.pdf",

        image: {

            type:
                "jpeg",

            quality:
                0.98
        },

        html2canvas: {

            scale:
                2,

            backgroundColor:
                "#0f0f12",

            useCORS:
                true
        },

        jsPDF: {

            unit:
                "in",

            format:
                "letter",

            orientation:
                "portrait"
        }
    };


    logToConsole(
        "reports",
        "Generating high-fidelity PDF export...",
        "log-step"
    );


    if (
        typeof html2pdf !==
        "function"
    ) {

        logToConsole(
            "reports",
            "html2pdf.js is not available.",
            "log-error"
        );

        return;
    }


    html2pdf()
        .set(opt)
        .from(element)
        .save()
        .then(
            () => {

                logToConsole(
                    "reports",
                    "PDF export successful.",
                    "log-success"
                );
            }
        );
}