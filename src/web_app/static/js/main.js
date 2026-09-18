// ============================================================
// POWER SYSTEM PROTECTION ML - INTERACTIVE FRONTEND
// ============================================================

// ============================================================
// HELPER FUNCTIONS
// ============================================================
function idmt_trip_time(I_primary, Is, TMS) {
  if (I_primary <= Is) return Infinity;
  const M = I_primary / Is;
  const t = TMS * 0.14 / (Math.pow(M, 0.02) - 1.0);
  return Math.max(0.04, t);
}

function distance_zone(Z_fault, Z_total) {
  const Z1 = Z_total * 0.25;
  const Z2 = Z_total * 0.50;
  const Z3 = Z_total * 0.75;
  if (Z_fault <= Z1) return {zone: 1, operates: true};
  if (Z_fault <= Z2) return {zone: 2, operates: true};
  if (Z_fault <= Z3) return {zone: 3, operates: true};
  return {zone: 0, operates: false};
}

// ============================================================
// IDMT RELAY FUNCTIONS
// ============================================================
function drawIDMTChart(r1TMS, r1Is, r2TMS, r2Is, faultI) {
  const Mrange = [];
  for (let i = 0; i <= 500; i++) Mrange.push(1.05 + i * 0.04);

  const curveR1 = Mrange.map(M => {
    const t = r1TMS * 0.14 / (Math.pow(M, 0.02) - 1);
    return {x: M, y: Math.min(t, 3)};
  });
  const curveR2 = Mrange.map(M => {
    const t = r2TMS * 0.14 / (Math.pow(M, 0.02) - 1);
    return {x: M, y: Math.min(t, 3)};
  });

  const Mfault = faultI / r1Is;
  const tR1 = idmt_trip_time(faultI, r1Is, r1TMS);
  const tR2 = idmt_trip_time(faultI, r2Is, r2TMS);
  const faultT1 = tR1 < Infinity ? tR1 : 3;
  const faultT2 = tR2 < Infinity ? tR2 : 3;

  const ctx = document.getElementById('idmtChart');
  if (window.idmtChartInstance) window.idmtChartInstance.destroy();

  window.idmtChartInstance = new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: [
        {label: 'R1 Operating Curve', data: curveR1, borderColor:'#1a73e8', backgroundColor:'rgba(26,115,232,0.1)', showLine:true, pointRadius:0, borderWidth:2.5},
        {label: 'R2 Operating Curve', data: curveR2, borderColor:'#d93025', backgroundColor:'rgba(217,48,37,0.1)', showLine:true, pointRadius:0, borderWidth:2.5},
        {label: 'R1 Trip Point', data: [{x:Mfault, y:faultT1}], borderColor:'#1a73e8', backgroundColor:'#1a73e8', pointRadius:8, pointStyle:'cross', showLine:false},
        {label: 'R2 Trip Point', data: [{x:faultI/r2Is, y:faultT2}], borderColor:'#d93025', backgroundColor:'#d93025', pointRadius:8, pointStyle:'cross', showLine:false},
        {label: 'Min Operating Time', data: Mrange.map(m=>({x:m, y:0.04})), borderColor:'#9aa0a6', borderDash:[5,5], showLine:true, pointRadius:0, borderWidth:1}
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {x: {type:'linear', title:{display:true, text:'Current Multiplier (M)'}, min:1, max:12}, y: {title:{display:true, text:'Operating Time (s)'}, min:0, max:3}},
      plugins: {legend: {position: 'bottom'}}
    }
  });
}

// ============================================================
// DISTANCE RELAY FUNCTIONS
// ============================================================
function drawDistanceChart(Ztotal, Zfault) {
  const Z1 = Ztotal * 0.25;
  const Z2 = Ztotal * 0.50;
  const Z3 = Ztotal * 0.75;
  const Z4 = Ztotal * 1.0;

  const ctx = document.getElementById('distanceChart');
  if (window.distanceChartInstance) window.distanceChartInstance.destroy();

  window.distanceChartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      datasets: [
        {label:'Zone 1 (0-25%)', data: [{x:0,y:0},{x:Z1,y:Z1}], borderColor:'#0f9d58', backgroundColor:'rgba(15,157,88,0.1)', showLine:true, pointRadius:0, borderWidth:3, fill:true},
        {label:'Zone 2 (25-50%)', data: [{x:Z1,y:Z1},{x:Z2,y:Z2}], borderColor:'#1a73e8', backgroundColor:'rgba(26,115,232,0.1)', showLine:true, pointRadius:0, borderWidth:3, fill:true},
        {label:'Zone 3 (50-75%)', data: [{x:Z2,y:Z2},{x:Z3,y:Z3}], borderColor:'#f9ab00', backgroundColor:'rgba(249,171,0,0.1)', showLine:true, pointRadius:0, borderWidth:3, fill:true},
        {label:'Fault Impedance', data: [{x:0,y:0},{x:Zfault,y:Zfault}], borderColor:'#d93025', backgroundColor:'#d93025', pointRadius:8, pointStyle:'star', showLine:true, borderWidth:2}
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {x: {type:'linear', title:{display:true, text:'Distance (%)'}, min:0, max:Z4*1.1}, y: {title:{display:true, text:'Impedance (Ω)'}, min:0, max:Z4*1.1}},
      plugins: {legend: {position: 'bottom'}}
    }
  });
}

// ============================================================
// DIFFERENTIAL RELAY FUNCTIONS
// ============================================================
function drawDiffChart(Iin, Iout, operates) {
  const ctx = document.getElementById('diffChart');
  if (window.diffChartInstance) window.diffChartInstance.destroy();

  const imbalance = Math.abs(Iin - Iout) / Math.max(Iin, Iout) * 100;

  window.diffChartInstance = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['I_in (A)', 'I_out (A)', 'Imbalance (%)'],
      datasets: [{
        label: 'Value',
        data: [Iin, Iout, imbalance],
        backgroundColor: [operates ? '#d93025' : '#0f9d58', operates ? '#d93025' : '#0f9d58', operates ? '#f9ab00' : '#5f6368'],
        borderColor: [operates ? '#d93025' : '#0f9d58', operates ? '#d93025' : '#0f9d58', operates ? '#f9ab00' : '#5f6368'],
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {y: {title: {display: true, text: 'Value'}}},
      plugins: {legend: {display: false}}
    }
  });
}

// ============================================================
// MAIN INITIALIZATION
// ============================================================
document.addEventListener('DOMContentLoaded', function() {
  // Initialize IDMT chart
  const r1TMS = 0.161, r1Is = 234.3, r2TMS = 0.05, r2Is = 203.7, faultI = 800;
  drawIDMTChart(r1TMS, r1Is, r2TMS, r2Is, faultI);

  // Initialize Distance chart
  const Ztotal = 40, Zfault = 14;
  drawDistanceChart(Ztotal, Zfault);

  // Initialize Differential chart
  const Iin = 2.5, Iout = 2.4, operates = false;
  drawDiffChart(Iin, Iout, operates);

  console.log('Power System Protection ML Playground initialized');
});