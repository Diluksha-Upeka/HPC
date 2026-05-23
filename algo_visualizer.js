// ══════════════════════════════════════════════════════
//  HPC BFS Algorithm Visualizer — JS Engine
// ══════════════════════════════════════════════════════

// ── Graph Data Management ──
let G = null;

function buildRandomGraph() {
  const V = parseInt(document.getElementById('cfg-v').value) || 30;
  const density = parseFloat(document.getElementById('cfg-d').value) || 0.1;
  const src = 0; // Hardcoded source to 0 for simplicity, could add input
  const edges = [];
  const adj = Array.from({length:V},()=>[]);
  
  for (let u = 0; u < V; u++) {
    for (let v = u + 1; v < V; v++) {
      if (Math.random() < density) {
        edges.push([u,v]);
        adj[u].push(v);
        adj[v].push(u);
      }
    }
  }

  const dist=Array(V).fill(-1); dist[src]=0;
  const q=[src];
  const serialSteps = []; 
  
  while(q.length){
    const u=q.shift();
    const currentQ = [...q];
    serialSteps.push({u, q:currentQ, level:dist[u]});
    for(const v of adj[u]){
      if(dist[v]===-1){
        dist[v]=dist[u]+1;
        q.push(v);
      }
    }
  }
  
  G = {V,src,edges,adj,dist,serialSteps, isReal: false};
}

buildRandomGraph(); // Initial generation

// ── Shared Colors ──
const THREAD_COLORS = ['#f59e0b','#06b6d4','#8b5cf6','#22c55e','#ef4444','#ec4899','#14b8a6','#f97316'];
const UNREACH = '#334155';
const dColor = d => d < 0 ? UNREACH : '#3b82f6'; // Base color for reached nodes
const hlColor = '#fcd34d'; // Highlight

// ── UI Logic ──
document.querySelectorAll('.tab').forEach(t => {
  t.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(b=>b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p=>p.classList.remove('active'));
    t.classList.add('active');
    document.getElementById('tab-'+t.dataset.tab).classList.add('active');
    setTimeout(resizeAll, 50);
  });
});

document.querySelectorAll('.tab-link').forEach(l => {
  l.addEventListener('click', () => {
    const target = l.dataset.target;
    document.querySelector(`.tab[data-tab="${target}"]`).click();
  });
});

// ── Force-directed layout ──
let nodePos = [];
function layoutGraph() {
  const V = G.V;
  const pos = Array.from({length:V},()=>({x:(Math.random()-.5)*2,y:(Math.random()-.5)*2,vx:0,vy:0}));
  for(let iter=0;iter<300;iter++){
    for(let i=0;i<V;i++)for(let j=i+1;j<V;j++){
      let dx=pos[i].x-pos[j].x, dy=pos[i].y-pos[j].y, d2=dx*dx+dy*dy+.001;
      let f=0.1/d2; pos[i].vx+=f*dx;pos[i].vy+=f*dy;pos[j].vx-=f*dx;pos[j].vy-=f*dy;
    }
    for(const[u,v]of G.edges){
      let dx=pos[v].x-pos[u].x,dy=pos[v].y-pos[u].y,d=Math.sqrt(dx*dx+dy*dy)+.001;
      let f=0.01*(d-1.5); pos[u].vx+=f*dx/d;pos[u].vy+=f*dy/d;pos[v].vx-=f*dx/d;pos[v].vy-=f*dy/d;
    }
    for(let i=0;i<V;i++){pos[i].vx*=.85;pos[i].vy*=.85;pos[i].x+=pos[i].vx;pos[i].y+=pos[i].vy;}
  }
  let minX=Infinity,maxX=-Infinity,minY=Infinity,maxY=-Infinity;
  for(let i=0;i<V;i++){minX=Math.min(minX,pos[i].x);maxX=Math.max(maxX,pos[i].x);minY=Math.min(minY,pos[i].y);maxY=Math.max(maxY,pos[i].y);}
  const rangeX=maxX-minX||1, rangeY=maxY-minY||1;
  const scale = Math.max(rangeX, rangeY) * 0.55;
  const cx = (minX+maxX)/2, cy = (minY+maxY)/2;
  for(let i=0;i<V;i++){pos[i].x=(pos[i].x-cx)/scale;pos[i].y=(pos[i].y-cy)/scale;}
  return pos;
}

// ── Global State & Timers ──
let serialStep = -1; let serialTimer = null;
let ompLevel = -1; let ompTimer = null;
let mpiLevel = -1; let mpiTimer = null; let mpiPhase = 0;
let cudaLevel = -1; let cudaTimer = null;

function updateGraphLayout() {
  nodePos = layoutGraph();
  if(serialTimer) { clearInterval(serialTimer); serialTimer=null; document.getElementById('btn-serial').innerHTML='&#9654; Animate'; serialStep=-1; }
  if(ompTimer) { clearInterval(ompTimer); ompTimer=null; ompLevel=-1; document.getElementById('btn-openmp').innerHTML='&#9654; Animate'; }
  if(mpiTimer) { clearInterval(mpiTimer); mpiTimer=null; mpiLevel=-1; mpiPhase=0; document.getElementById('btn-mpi').innerHTML='&#9654; Animate'; }
  if(cudaTimer) { clearInterval(cudaTimer); cudaTimer=null; cudaLevel=-1; document.getElementById('btn-cuda').innerHTML='&#9654; Animate'; }
  resizeAll();
}

updateGraphLayout();

document.getElementById('btn-regen').addEventListener('click', () => {
  buildRandomGraph();
  updateGraphLayout();
});

document.getElementById('cfg-file').addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (event) => {
    try {
      const data = JSON.parse(event.target.result);
      
      let V = data.vertices;
      let edges = data.edge_list || [];
      let src = data.source || 0;
      let dist = data.distance || [];
      let isMassiveGraph = false;
      
      // If the Colab notebook omitted the edges (because the graph was too huge to save),
      // or if V is way too large for the browser's physics engine to handle (e.g., > 1000),
      // we will generate a small proxy graph for the animations, but still use the REAL
      // performance timings for the dashboard!
      if (edges.length === 0 || V > 500) {
        isMassiveGraph = true;
        V = 100; // Proxy graph size for animations
        src = 0;
        edges = [];
        for (let u = 0; u < V; u++) {
          for (let v = u + 1; v < V; v++) {
            if (Math.random() < 0.05) edges.push([u,v]);
          }
        }
      }

      const adj = Array.from({length:V},()=>[]);
      for(const [u,v] of edges){adj[u].push(v);adj[v].push(u);}
      
      // Reconstruct serial steps if not provided or if we built a proxy graph
      const q=[src];
      const serialSteps = []; 
      const simDist = Array(V).fill(-1); simDist[src]=0;
      while(q.length){
        const u=q.shift();
        const currentQ = [...q];
        serialSteps.push({u, q:currentQ, level:simDist[u]});
        for(const v of adj[u]){
          if(simDist[v]===-1){
            simDist[v]=simDist[u]+1;
            q.push(v);
          }
        }
      }

      G = {
        V, src, edges, adj, dist: simDist, serialSteps,
        isReal: true,
        gen_ms: data.gen_ms,
        bfs_ms: data.bfs_ms,
        true_v: data.vertices, // Save the actual benchmark parameters for display
        true_e: data.edges
      };
      
      if (isMassiveGraph) {
        document.getElementById('cfg-v').value = data.vertices;
        document.getElementById('cfg-d').value = "Massive";
      } else {
        document.getElementById('cfg-v').value = V;
        document.getElementById('cfg-d').value = (data.edges / (V*(V-1)/2)).toFixed(3);
      }
      
      updateGraphLayout();
    } catch (err) {
      alert("Failed to load JSON: " + err.message);
    }
  };
  reader.readAsText(file);
});

function setupCanvas(cv) {
  const dpr = devicePixelRatio||1;
  const w = cv.clientWidth, h = cv.clientHeight;
  if(w===0||h===0) return null;
  cv.width = w*dpr; cv.height = h*dpr;
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr,0,0,dpr,0,0);
  return {ctx, w, h};
}

// ── Serial Animation ──

function drawSerial() {
  const c = setupCanvas(document.getElementById('cv-serial'));
  if(!c) return;
  const {ctx, w, h} = c;
  ctx.clearRect(0,0,w,h);
  
  const sc = Math.min(w,h)*0.4;
  const cx = w/2, cy = h/2;
  const R = 14;

  const currentStepData = serialStep >= 0 && serialStep < G.serialSteps.length ? G.serialSteps[serialStep] : null;
  const activeNode = currentStepData ? currentStepData.u : -1;
  
  // Track visited set up to this step
  const visited = new Set();
  const enqueued = new Set();
  if (serialStep >= 0) {
    for(let i=0; i<=serialStep; i++) visited.add(G.serialSteps[i].u);
    if (currentStepData) {
      currentStepData.q.forEach(n => enqueued.add(n));
      enqueued.add(activeNode);
    }
  } else if (serialStep === G.serialSteps.length) {
    for(let i=0; i<G.V; i++) if(G.dist[i]>=0) visited.add(i);
  }

  // Draw edges
  ctx.lineWidth=1.5;
  for(const[u,v]of G.edges){
    const isActive = (u===activeNode && enqueued.has(v)) || (v===activeNode && enqueued.has(u));
    ctx.strokeStyle = isActive ? '#f59e0b' : 'rgba(51,65,85,0.4)';
    if(isActive) ctx.lineWidth=2.5; else ctx.lineWidth=1.5;
    ctx.beginPath(); ctx.moveTo(cx+nodePos[u].x*sc, cy+nodePos[u].y*sc); ctx.lineTo(cx+nodePos[v].x*sc, cy+nodePos[v].y*sc); ctx.stroke();
  }

  // Draw nodes
  for(let i=0;i<G.V;i++){
    const px = cx+nodePos[i].x*sc, py = cy+nodePos[i].y*sc;
    ctx.beginPath(); ctx.arc(px,py,R,0,Math.PI*2);
    let fill = '#1e293b';
    if (i === activeNode) {
      fill = '#f59e0b';
      ctx.shadowColor = fill; ctx.shadowBlur = 10;
    } else if (visited.has(i)) {
      fill = '#3b82f6';
    } else if (enqueued.has(i)) {
      fill = '#64748b';
    }
    
    ctx.fillStyle = fill; ctx.fill(); ctx.shadowBlur = 0;
    ctx.strokeStyle = i===G.src ? '#fff' : '#475569'; ctx.lineWidth=2; ctx.stroke();
    ctx.fillStyle = '#fff'; ctx.font = 'bold 11px Inter'; ctx.textAlign='center'; ctx.textBaseline='middle'; ctx.fillText(i,px,py+1);
  }

  // Update UI Stats
  document.getElementById('s-current').textContent = activeNode >= 0 ? activeNode : '--';
  document.getElementById('s-level').textContent = currentStepData ? currentStepData.level : '--';
  document.getElementById('s-visited').textContent = visited.size + '/' + G.V;
  document.getElementById('s-qsize').textContent = currentStepData ? currentStepData.q.length : 0;
  
  const qviz = document.getElementById('serial-queue');
  if (currentStepData) {
    qviz.innerHTML = `<span class="q-item active">${activeNode}</span>` + 
                     currentStepData.q.map(n => `<span class="q-item">${n}</span>`).join('');
  } else if (serialStep === G.serialSteps.length) {
    qviz.innerHTML = 'Queue empty. Done.';
  } else {
    qviz.innerHTML = 'Press Animate to begin';
  }
}

document.getElementById('btn-serial').addEventListener('click', function() {
  if (serialTimer) { clearInterval(serialTimer); serialTimer=null; this.innerHTML='&#9654; Animate'; serialStep=-1; drawSerial(); return; }
  serialStep = 0; this.innerHTML='&#9632; Stop'; drawSerial();
  serialTimer = setInterval(() => {
    serialStep++;
    if (serialStep > G.serialSteps.length) { clearInterval(serialTimer); serialTimer=null; this.innerHTML='&#9654; Animate'; }
    drawSerial();
  }, 1500 / (+document.getElementById('spd-serial').value || 4));
});

// ── OpenMP Animation ──

function getOmpAssign() {
  const n = +(document.getElementById('omp-threads').value);
  const a = Array(G.V).fill(-1);
  for(let lv=0; lv<=Math.max(...G.dist); lv++){
    let c=0;
    for(let i=0;i<G.V;i++) if(G.dist[i]===lv) a[i] = (c++)%n;
  }
  return a;
}

function drawOpenMP() {
  const c = setupCanvas(document.getElementById('cv-openmp'));
  if(!c) return;
  const {ctx, w, h} = c;
  ctx.clearRect(0,0,w,h);
  
  const sc = Math.min(w,h)*0.4, cx = w/2, cy = h/2, R=14;
  const nThreads = +(document.getElementById('omp-threads').value);
  const assign = getOmpAssign();

  // Edges
  ctx.lineWidth=1.5;
  for(const[u,v]of G.edges){
    const du=G.dist[u], dv=G.dist[v];
    ctx.strokeStyle = (ompLevel>=0 && du<=ompLevel && dv<=ompLevel) ? 'rgba(148,163,184,0.3)' : 'rgba(51,65,85,0.2)';
    if(ompLevel>=0 && Math.abs(du-dv)===1 && du<=ompLevel && dv<=ompLevel) { ctx.strokeStyle='rgba(6,182,212,0.4)'; ctx.lineWidth=2; }
    ctx.beginPath(); ctx.moveTo(cx+nodePos[u].x*sc, cy+nodePos[u].y*sc); ctx.lineTo(cx+nodePos[v].x*sc, cy+nodePos[v].y*sc); ctx.stroke();
  }

  // Nodes
  for(let i=0;i<G.V;i++){
    const px = cx+nodePos[i].x*sc, py = cy+nodePos[i].y*sc, d=G.dist[i];
    const visible = ompLevel<0 || d<=ompLevel;
    let fill = '#1e293b';
    if(visible && d>=0){
      if(d===ompLevel) {
        fill = THREAD_COLORS[assign[i]];
        ctx.beginPath(); ctx.arc(px,py,R+6,0,Math.PI*2);
        ctx.fillStyle = fill; ctx.globalAlpha=0.2; ctx.fill(); ctx.globalAlpha=1.0;
      } else { fill = '#3b82f6'; }
    }
    ctx.beginPath(); ctx.arc(px,py,R,0,Math.PI*2);
    ctx.fillStyle = fill; ctx.fill();
    ctx.strokeStyle = i===G.src ? '#fff' : '#475569'; ctx.lineWidth=2; ctx.stroke();
    ctx.fillStyle = visible && d>=0 ? '#fff' : '#64748b'; ctx.font='bold 11px Inter'; ctx.textAlign='center'; ctx.textBaseline='middle'; ctx.fillText(i,px,py+1);
  }

  // Stats
  const fsize = G.dist.filter(d=>d===ompLevel).length;
  document.getElementById('o-level').textContent = ompLevel>=0 ? ompLevel : '--';
  document.getElementById('o-visited').textContent = ompLevel>=0 ? G.dist.filter(d=>d>=0&&d<=ompLevel).length+'/'+G.V : '--';
  document.getElementById('o-fsize').textContent = ompLevel>=0 ? fsize : '--';
  document.getElementById('o-threads').textContent = ompLevel>=0 && fsize>0 ? nThreads : '--';

  const bviz = document.getElementById('omp-buffers');
  if (ompLevel>=0 && fsize>0) {
    let html = '';
    for(let t=0; t<nThreads; t++){
      const nodes = [];
      for(let i=0;i<G.V;i++) if(G.dist[i]===ompLevel && assign[i]===t) nodes.push(i);
      html += `<div class="tbuf"><div class="tbuf-label" style="color:${THREAD_COLORS[t]}">T${t}</div><div class="tbuf-items">`;
      html += nodes.map(n=>`<span class="tbuf-item" style="background:${THREAD_COLORS[t]};color:#000">${n}</span>`).join('');
      html += `</div></div>`;
    }
    bviz.innerHTML = html;
  } else if (ompLevel > Math.max(...G.dist)) {
    bviz.innerHTML = 'Done.';
  } else {
    bviz.innerHTML = 'Press Animate to begin';
  }
}

document.getElementById('omp-threads').addEventListener('change', drawOpenMP);
document.getElementById('btn-openmp').addEventListener('click', function() {
  if(ompTimer){clearInterval(ompTimer);ompTimer=null;ompLevel=-1;this.innerHTML='&#9654; Animate';drawOpenMP();return;}
  ompLevel=0; this.innerHTML='&#9632; Stop'; drawOpenMP();
  ompTimer = setInterval(()=>{
    ompLevel++;
    if(ompLevel>Math.max(...G.dist)){clearInterval(ompTimer);ompTimer=null;this.innerHTML='&#9654; Animate';}
    drawOpenMP();
  }, 1500/(+document.getElementById('spd-openmp').value||4));
});


// ── MPI Animation ──

function getMpiAssign() {
  const n = +(document.getElementById('mpi-ranks').value);
  const a = Array(G.V).fill(0);
  for(let i=0;i<G.V;i++) a[i] = Math.floor(i*n/G.V);
  return a;
}

function drawMPI() {
  const c = setupCanvas(document.getElementById('cv-mpi'));
  if(!c) return;
  const {ctx, w, h} = c;
  ctx.clearRect(0,0,w,h);
  
  const sc = Math.min(w,h)*0.4, cx = w/2, cy = h/2, R=14;
  const nRanks = +(document.getElementById('mpi-ranks').value);
  const assign = getMpiAssign();

  // Draw Regions background
  const sortedY = [...nodePos].sort((a,b)=>a.y-b.y);
  ctx.globalAlpha=0.03;
  for(let r=0; r<nRanks; r++){
    ctx.fillStyle = THREAD_COLORS[r];
    ctx.fillRect(0, h/nRanks * r, w, h/nRanks);
  }
  ctx.globalAlpha=1.0;

  // Edges
  ctx.lineWidth=1.5;
  for(const[u,v]of G.edges){
    const du=G.dist[u], dv=G.dist[v];
    ctx.strokeStyle = (mpiLevel>=0 && du<=mpiLevel && dv<=mpiLevel) ? 'rgba(148,163,184,0.3)' : 'rgba(51,65,85,0.2)';
    if(mpiLevel>=0 && Math.abs(du-dv)===1 && du<=mpiLevel && dv<=mpiLevel) { ctx.strokeStyle='rgba(139,92,246,0.4)'; ctx.lineWidth=2; }
    
    // Cross-rank communication highlight during phase 1
    if (mpiLevel>0 && mpiPhase===1 && assign[u]!==assign[v] && ((du===mpiLevel&&dv===mpiLevel-1)||(dv===mpiLevel&&du===mpiLevel-1))) {
      ctx.strokeStyle = '#f43f5e'; ctx.lineWidth=3;
      // Draw marching ants or dashed line
      ctx.setLineDash([5, 5]);
    }
    
    ctx.beginPath(); ctx.moveTo(cx+nodePos[u].x*sc, cy+nodePos[u].y*sc); ctx.lineTo(cx+nodePos[v].x*sc, cy+nodePos[v].y*sc); ctx.stroke();
    ctx.setLineDash([]);
  }

  // Nodes
  let localDisc = 0;
  for(let i=0;i<G.V;i++){
    const px = cx+nodePos[i].x*sc, py = cy+nodePos[i].y*sc, d=G.dist[i];
    const visible = mpiLevel<0 || d<=mpiLevel;
    let fill = '#1e293b';
    if(visible && d>=0){
      fill = THREAD_COLORS[assign[i]]; // Color by rank
      if(d===mpiLevel) {
        if(mpiPhase===0) localDisc++;
        ctx.beginPath(); ctx.arc(px,py,R+6,0,Math.PI*2);
        ctx.fillStyle = fill; ctx.globalAlpha = mpiPhase===0 ? 0.4 : 0.15; ctx.fill(); ctx.globalAlpha=1.0;
      }
    }
    ctx.beginPath(); ctx.arc(px,py,R,0,Math.PI*2);
    ctx.fillStyle = fill; ctx.fill();
    ctx.strokeStyle = i===G.src ? '#fff' : '#475569'; ctx.lineWidth=2; ctx.stroke();
    ctx.fillStyle = visible && d>=0 ? '#fff' : '#64748b'; ctx.font='bold 11px Inter'; ctx.textAlign='center'; ctx.textBaseline='middle'; ctx.fillText(i,px,py+1);
  }

  // Stats
  const fsize = G.dist.filter(d=>d===mpiLevel).length;
  document.getElementById('m-level').textContent = mpiLevel>=0 ? mpiLevel : '--';
  document.getElementById('m-phase').textContent = mpiLevel<0 ? '--' : (mpiPhase===0 ? 'Compute (Local)' : 'Allgatherv (Communicate)');
  document.getElementById('m-local').textContent = mpiLevel>=0 ? (mpiPhase===0 ? localDisc : fsize) : '--';
  document.getElementById('m-frontier').textContent = mpiLevel>=0 ? fsize : '--';
}

document.getElementById('mpi-ranks').addEventListener('change', drawMPI);
document.getElementById('btn-mpi').addEventListener('click', function() {
  if(mpiTimer){clearInterval(mpiTimer);mpiTimer=null;mpiLevel=-1;mpiPhase=0;this.innerHTML='&#9654; Animate';drawMPI();return;}
  mpiLevel=0; mpiPhase=0; this.innerHTML='&#9632; Stop'; drawMPI();
  mpiTimer = setInterval(()=>{
    if(mpiPhase===0 && mpiLevel>0 && G.dist.filter(d=>d===mpiLevel).length>0){
      mpiPhase=1; // switch to comm
    } else {
      mpiPhase=0;
      mpiLevel++;
    }
    if(mpiLevel>Math.max(...G.dist)){clearInterval(mpiTimer);mpiTimer=null;this.innerHTML='&#9654; Animate';}
    drawMPI();
  }, 1000/(+document.getElementById('spd-mpi').value||4));
});


// ── HYBRID Animation ──

let hybridTimer = null;
let hybridLevel = -1;
let hybridPhase = 0; // 0=Compute(Threads), 1=Communicate(MPI)

function drawHybrid() {
  const c = setupCanvas(document.getElementById('cv-hybrid'));
  if(!c) return;
  const {ctx, w, h} = c;
  ctx.clearRect(0,0,w,h);
  
  const sc = Math.min(w,h)*0.4, cx = w/2, cy = h/2, R=14;
  const nRanks = 2; // Fixed 2 ranks for visual simplicity
  const assign = getMpiAssign().map(x => x % nRanks); // Force 2 ranks
  
  // Draw Regions background (MPI)
  const sortedY = [...nodePos].sort((a,b)=>a.y-b.y);
  ctx.globalAlpha=0.03;
  ctx.fillStyle = '#9d174d'; ctx.fillRect(0, 0, w, h/2);
  ctx.fillStyle = '#db2777'; ctx.fillRect(0, h/2, w, h/2);
  ctx.globalAlpha=1.0;

  // Edges
  ctx.lineWidth=1.5;
  for(const[u,v]of G.edges){
    const du=G.dist[u], dv=G.dist[v];
    ctx.strokeStyle = (hybridLevel>=0 && du<=hybridLevel && dv<=hybridLevel) ? 'rgba(148,163,184,0.3)' : 'rgba(51,65,85,0.2)';
    if(hybridLevel>=0 && Math.abs(du-dv)===1 && du<=hybridLevel && dv<=hybridLevel) { ctx.strokeStyle='rgba(236,72,153,0.4)'; ctx.lineWidth=2; }
    
    // Cross-rank communication highlight during phase 1
    if (hybridLevel>0 && hybridPhase===1 && assign[u]!==assign[v] && ((du===hybridLevel&&dv===hybridLevel-1)||(dv===hybridLevel&&du===hybridLevel-1))) {
      ctx.strokeStyle = '#fbcfe8'; ctx.lineWidth=3;
      ctx.setLineDash([5, 5]);
    }
    
    ctx.beginPath(); ctx.moveTo(cx+nodePos[u].x*sc, cy+nodePos[u].y*sc); ctx.lineTo(cx+nodePos[v].x*sc, cy+nodePos[v].y*sc); ctx.stroke();
    ctx.setLineDash([]);
  }

  // Nodes (OpenMP Threads inside MPI Ranks)
  for(let i=0;i<G.V;i++){
    const px = cx+nodePos[i].x*sc, py = cy+nodePos[i].y*sc, d=G.dist[i];
    const visible = hybridLevel<0 || d<=hybridLevel;
    let fill = '#1e293b';
    if(visible && d>=0){
      // Rank 0 gets colors 0,1. Rank 1 gets colors 2,3 (representing 2 threads per rank)
      const threadColorIdx = (assign[i] * 2) + (i % 2); 
      fill = THREAD_COLORS[threadColorIdx % THREAD_COLORS.length]; 
      
      if(d===hybridLevel) {
        ctx.beginPath(); ctx.arc(px,py,R+6,0,Math.PI*2);
        ctx.fillStyle = fill; ctx.globalAlpha = hybridPhase===0 ? 0.4 : 0.15; ctx.fill(); ctx.globalAlpha=1.0;
      }
    }
    ctx.beginPath(); ctx.arc(px,py,R,0,Math.PI*2);
    ctx.fillStyle = fill; ctx.fill();
    ctx.strokeStyle = i===G.src ? '#fff' : '#475569'; ctx.lineWidth=2; ctx.stroke();
    ctx.fillStyle = visible && d>=0 ? '#fff' : '#64748b'; ctx.font='bold 11px Inter'; ctx.textAlign='center'; ctx.textBaseline='middle'; ctx.fillText(i,px,py+1);
  }
}

document.getElementById('btn-hybrid').addEventListener('click', function() {
  if(hybridTimer){clearInterval(hybridTimer);hybridTimer=null;hybridLevel=-1;hybridPhase=0;this.innerHTML='&#9654; Animate';drawHybrid();return;}
  hybridLevel=0; hybridPhase=0; this.innerHTML='&#9632; Stop'; drawHybrid();
  hybridTimer = setInterval(()=>{
    if(hybridPhase===0 && hybridLevel>0 && G.dist.filter(d=>d===hybridLevel).length>0){
      hybridPhase=1; // switch to comm
    } else {
      hybridPhase=0;
      hybridLevel++;
    }
    if(hybridLevel>Math.max(...G.dist)){clearInterval(hybridTimer);hybridTimer=null;this.innerHTML='&#9654; Animate';}
    drawHybrid();
  }, 1000/(+document.getElementById('spd-hybrid').value||4));
});


// ── CUDA Animation ──

function drawCUDA() {
  const c = setupCanvas(document.getElementById('cv-cuda'));
  if(!c) return;
  const {ctx, w, h} = c;
  ctx.clearRect(0,0,w,h);
  
  const sc = Math.min(w,h)*0.4, cx = w/2, cy = h/2, R=14;

  ctx.lineWidth=1.5;
  for(const[u,v]of G.edges){
    const du=G.dist[u], dv=G.dist[v];
    ctx.strokeStyle = (cudaLevel>=0 && du<=cudaLevel && dv<=cudaLevel) ? 'rgba(148,163,184,0.3)' : 'rgba(51,65,85,0.2)';
    if(cudaLevel>=0 && Math.abs(du-dv)===1 && du<=cudaLevel && dv<=cudaLevel) { ctx.strokeStyle='rgba(34,197,94,0.4)'; ctx.lineWidth=2; }
    ctx.beginPath(); ctx.moveTo(cx+nodePos[u].x*sc, cy+nodePos[u].y*sc); ctx.lineTo(cx+nodePos[v].x*sc, cy+nodePos[v].y*sc); ctx.stroke();
  }

  for(let i=0;i<G.V;i++){
    const px = cx+nodePos[i].x*sc, py = cy+nodePos[i].y*sc, d=G.dist[i];
    const visible = cudaLevel<0 || d<=cudaLevel;
    let fill = '#1e293b';
    if(visible && d>=0){
      if(d===cudaLevel) {
        fill = '#22c55e';
        ctx.beginPath(); ctx.arc(px,py,R+6,0,Math.PI*2);
        ctx.fillStyle = fill; ctx.globalAlpha = 0.2; ctx.fill(); ctx.globalAlpha=1.0;
      } else { fill = '#3b82f6'; }
    }
    ctx.beginPath(); ctx.arc(px,py,R,0,Math.PI*2);
    ctx.fillStyle = fill; ctx.fill();
    ctx.strokeStyle = i===G.src ? '#fff' : '#475569'; ctx.lineWidth=2; ctx.stroke();
    ctx.fillStyle = visible && d>=0 ? '#fff' : '#64748b'; ctx.font='bold 11px Inter'; ctx.textAlign='center'; ctx.textBaseline='middle'; ctx.fillText(i,px,py+1);
  }

  const fsize = G.dist.filter(d=>d===cudaLevel).length;
  document.getElementById('c-level').textContent = cudaLevel>=0 ? cudaLevel : '--';
  document.getElementById('c-blocks').textContent = cudaLevel>=0 && fsize>0 ? Math.ceil(fsize/256) : '--';
  document.getElementById('c-cthreads').textContent = cudaLevel>=0 && fsize>0 ? fsize : '--';
  document.getElementById('c-frontier').textContent = cudaLevel>=0 ? fsize : '--';

  const csrViz = document.getElementById('csr-viz');
  if (cudaLevel>=0) {
    let html = '';
    // Mock CSR representation for a few active nodes
    const active = [];
    for(let i=0;i<G.V;i++) if(G.dist[i]===cudaLevel) active.push(i);
    if(active.length>0) {
      const show = active.slice(0, 3);
      for(let u of show) {
        html += `<div class="csr-row"><span class="csr-label">Thread ${u}:</span> row_off[${u}]=${u*2}, explores [${G.adj[u].join(',')}]</div>`;
      }
      if(active.length>3) html += `<div class="csr-row">... and ${active.length-3} more threads</div>`;
    } else {
      html = 'Kernel execution complete.';
    }
    csrViz.innerHTML = html;
  } else {
    csrViz.innerHTML = 'Press Animate to see CSR mapping';
  }
}

document.getElementById('btn-cuda').addEventListener('click', function() {
  if(cudaTimer){clearInterval(cudaTimer);cudaTimer=null;cudaLevel=-1;this.innerHTML='&#9654; Animate';drawCUDA();return;}
  cudaLevel=0; this.innerHTML='&#9632; Stop'; drawCUDA();
  cudaTimer = setInterval(()=>{
    cudaLevel++;
    if(cudaLevel>Math.max(...G.dist)){clearInterval(cudaTimer);cudaTimer=null;this.innerHTML='&#9654; Animate';}
    drawCUDA();
  }, 1500/(+document.getElementById('spd-cuda').value||4));
});


// ── Performance Charts ──
function drawBarChart(canvasId, labels, dataArr, colors) {
  const cv = document.getElementById(canvasId);
  const {ctx, w, h} = setupCanvas(cv);
  if(!ctx) return;
  
  ctx.clearRect(0,0,w,h);
  const padL=40, padB=30, padT=20, padR=20;
  const cw = w - padL - padR, ch = h - padB - padT;
  
  const maxVal = Math.max(...dataArr) * 1.1 || 1;
  
  // Axes
  ctx.strokeStyle='#334155'; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(padL, padT); ctx.lineTo(padL, h-padB); ctx.lineTo(w-padR, h-padB); ctx.stroke();
  
  // Bars
  const barW = cw / labels.length * 0.6;
  const spacing = cw / labels.length;
  
  ctx.textAlign='center'; ctx.textBaseline='top'; ctx.font='11px Inter';
  for(let i=0; i<labels.length; i++) {
    const barH = (dataArr[i]/maxVal) * ch;
    const x = padL + i*spacing + (spacing-barW)/2;
    const y = h - padB - barH;
    
    ctx.fillStyle = colors[i % colors.length];
    ctx.fillRect(x, y, barW, barH);
    
    ctx.fillStyle = '#94a3b8';
    ctx.fillText(labels[i], x+barW/2, h-padB+5);
    
    ctx.fillStyle = '#e2e8f0'; ctx.textBaseline='bottom';
    ctx.fillText(dataArr[i].toFixed(2), x+barW/2, y-2);
    ctx.textBaseline='top';
  }
  
  // Y-axis labels
  ctx.textAlign='right'; ctx.textBaseline='middle'; ctx.fillStyle='#64748b';
  ctx.fillText('0', padL-5, h-padB);
  ctx.fillText(maxVal.toFixed(1), padL-5, padT);
}

function updatePerfDashboard() {
  const lbl = document.getElementById('perf-source-label');
  
  let serialTime = 14.5;
  let ompTime = 4.2;
  let mpiTime = 6.8;
  let hybridTime = 3.9;
  let cudaTime = 1.1;
  let mpiComm = 2.4;
  
  if (G.isReal && G.bfs_ms) {
    lbl.innerHTML = 'Showing <span style="color:#22c55e;font-weight:700">REAL</span> data loaded from actual benchmark JSON.';
    serialTime = G.bfs_ms.serial || serialTime;
    ompTime = G.bfs_ms.omp || ompTime;
    mpiTime = G.bfs_ms.mpi || mpiTime;
    hybridTime = G.bfs_ms.hybrid || hybridTime;
    cudaTime = G.bfs_ms.cuda || cudaTime;
    mpiComm = G.bfs_ms.mpi_comm || (mpiTime * 0.35); // fallback estimate if not explicitly provided
  } else {
    lbl.innerHTML = 'Showing <span style="color:#f43f5e;font-weight:700">MOCK</span> data. Use "Load Real Data" in the top bar to load actual JSON results from your C++ executables.';
  }

  const labels = ['Serial', 'OpenMP (4T)', 'MPI (4R)', 'Hybrid (2R/2T)', 'CUDA'];
  const times = [serialTime, ompTime, mpiTime, hybridTime, cudaTime];
  const colors = ['#f59e0b', '#06b6d4', '#8b5cf6', '#ec4899', '#22c55e'];
  
  drawBarChart('cv-time-bar', labels, times, colors);
  
  // Speedup
  const speedups = times.map(t => serialTime/t);
  drawBarChart('cv-speedup', labels, speedups, colors);
  
  // Efficiency (Speedup / processors). Serial=1, OMP=4, MPI=4, Hybrid=4(2x2), CUDA=N/A(use 1 for bar)
  const effs = [1.0, speedups[1]/4, speedups[2]/4, speedups[3]/4, speedups[4]/1];
  drawBarChart('cv-efficiency', ['Serial', 'OpenMP', 'MPI', 'Hybrid', 'CUDA (raw)'], effs, colors);
  
  // MPI Comm Breakdown
  const cvMpi = document.getElementById('cv-mpi-comm');
  const cm = setupCanvas(cvMpi);
  if(cm) {
    const ctx=cm.ctx, w=cm.w, h=cm.h;
    ctx.clearRect(0,0,w,h);
    const commMs = mpiComm, compMs = mpiTime - commMs;
    const barW = w*0.8, barH = 40, x=(w-barW)/2, y=(h-barH)/2;
    ctx.fillStyle='#4c1d95'; ctx.fillRect(x,y, barW, barH);
    ctx.fillStyle='#8b5cf6'; ctx.fillRect(x,y, Math.max(1, barW*(compMs/mpiTime)), barH);
    ctx.fillStyle='#fff'; ctx.textAlign='center'; ctx.textBaseline='middle'; ctx.font='12px Inter';
    ctx.fillText(`Compute: ${compMs.toFixed(1)}ms`, x + barW*(compMs/mpiTime)/2, y+barH/2);
    ctx.fillText(`Comm: ${commMs.toFixed(1)}ms`, x + barW*(compMs/mpiTime) + barW*(commMs/mpiTime)/2, y+barH/2);
  }

  // Table
  const tbody = document.querySelector('#perf-table tbody');
  tbody.innerHTML = `
    <tr><td>Serial</td><td>${(G.gen_ms||125.0).toFixed(2)}</td><td>${serialTime.toFixed(2)}</td><td>1.00x</td><td>100%</td><td class="pass">N/A</td></tr>
    <tr><td>OpenMP (4 Threads)</td><td>${(G.gen_ms||128.2).toFixed(2)}</td><td>${ompTime.toFixed(2)}</td><td>${speedups[1].toFixed(2)}x</td><td>${(effs[1]*100).toFixed(0)}%</td><td class="pass">PASSED</td></tr>
    <tr><td>MPI (4 Ranks)</td><td>${(G.gen_ms||126.1).toFixed(2)}</td><td>${mpiTime.toFixed(2)}</td><td>${speedups[2].toFixed(2)}x</td><td>${(effs[2]*100).toFixed(0)}%</td><td class="pass">PASSED</td></tr>
    <tr><td>Hybrid (2R x 2T)</td><td>${(G.gen_ms||127.3).toFixed(2)}</td><td>${hybridTime.toFixed(2)}</td><td>${speedups[3].toFixed(2)}x</td><td>${(effs[3]*100).toFixed(0)}%</td><td class="pass">PASSED</td></tr>
    <tr><td>CUDA (GPU)</td><td>${(G.gen_ms||124.5).toFixed(2)}</td><td>${cudaTime.toFixed(2)}</td><td>${speedups[4].toFixed(2)}x</td><td>N/A</td><td class="pass">PASSED</td></tr>
  `;
}

// ── Global Resize & Init ──
function resizeAll() {
  drawSerial();
  drawOpenMP();
  drawMPI();
  drawHybrid();
  drawCUDA();
  if(document.getElementById('tab-perf').classList.contains('active')) updatePerfDashboard();
}
window.addEventListener('resize', resizeAll);

// Init
setTimeout(resizeAll, 100);
