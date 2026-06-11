/* ─────────────────────────────────────────────────────────────────────────
 *  Dashboard logic: tab switching, upload preview, webcam record, submit.
 * ───────────────────────────────────────────────────────────────────────── */

(() => {
  const $ = (sel) => document.querySelector(sel);
  const tabs = document.querySelectorAll('.tab');
  const panels = document.querySelectorAll('.tab-panel');
  const sourceField  = $('#sourceField');
  const btnPredict   = $('#btnPredict');
  const videoInput   = $('#videoInput');
  const uploadPrev   = $('#uploadPreview');

  // ─── Tab switching ─────────────────────────────────────────────────────
  let activeTab = 'upload';
  tabs.forEach((t) => t.addEventListener('click', () => {
    tabs.forEach((x) => x.classList.remove('tab-active'));
    t.classList.add('tab-active');
    activeTab = t.dataset.tab;
    sourceField.value = activeTab;
    panels.forEach((p) => {
      p.classList.toggle('hidden', p.dataset.panel !== activeTab);
    });
    refreshPredictBtn();
  }));

  // ─── Upload preview ────────────────────────────────────────────────────
  videoInput?.addEventListener('change', () => {
    const f = videoInput.files[0];
    if (!f) { uploadPrev.hidden = true; refreshPredictBtn(); return; }
    uploadPrev.src = URL.createObjectURL(f);
    uploadPrev.hidden = false;
    refreshPredictBtn();
  });

  // ─── Webcam recording ──────────────────────────────────────────────────
  const liveCam       = $('#liveCam');
  const btnStartCam   = $('#btnStartCam');
  const btnRecord     = $('#btnRecord');
  const btnStop       = $('#btnStop');
  const recTimer      = $('#recTimer');
  const recordedPrev  = $('#recordedPreview');

  let mediaStream = null;
  let mediaRecorder = null;
  let recordedChunks = [];
  let recordedBlob = null;
  let timerInterval = null;

  btnStartCam?.addEventListener('click', async () => {
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: 640, height: 480 },
        audio: false,
      });
      liveCam.srcObject = mediaStream;
      btnRecord.disabled = false;
      btnStartCam.disabled = true;
      btnStartCam.textContent = '✓ Kamera aktif';
    } catch (err) {
      alert('Tidak bisa akses kamera: ' + err.message);
    }
  });

  btnRecord?.addEventListener('click', () => {
    if (!mediaStream) return;
    recordedChunks = [];

    // pilih mime type yang didukung browser
    const candidates = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'];
    const mimeType = candidates.find((m) => MediaRecorder.isTypeSupported(m)) || 'video/webm';
    mediaRecorder = new MediaRecorder(mediaStream, { mimeType });

    mediaRecorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) recordedChunks.push(e.data);
    };
    mediaRecorder.onstop = () => {
      recordedBlob = new Blob(recordedChunks, { type: 'video/webm' });
      recordedPrev.src = URL.createObjectURL(recordedBlob);
      recordedPrev.hidden = false;
      refreshPredictBtn();
    };

    mediaRecorder.start();
    btnRecord.disabled = true;
    btnStop.disabled = false;

    const start = Date.now();
    timerInterval = setInterval(() => {
      const s = Math.floor((Date.now() - start) / 1000);
      recTimer.textContent = `${String(Math.floor(s/60)).padStart(2,'0')}:${String(s%60).padStart(2,'0')}`;
    }, 200);
  });

  btnStop?.addEventListener('click', () => {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
      mediaRecorder.stop();
    }
    clearInterval(timerInterval);
    btnRecord.disabled = false;
    btnStop.disabled = true;
  });

  // ─── Submit ────────────────────────────────────────────────────────────
  const form        = $('#predictForm');
  const statusBox   = $('#status');
  const resultBox   = $('#resultBox');
  const traitGrid   = $('#traitGrid');
  const detailLink  = $('#detailLink');

  function refreshPredictBtn() {
    let ok = false;
    if (activeTab === 'upload') ok = !!(videoInput && videoInput.files[0]);
    if (activeTab === 'record') ok = !!recordedBlob;
    btnPredict.disabled = !ok;
  }

  form?.addEventListener('submit', async (e) => {
    e.preventDefault();

    const fd = new FormData();
    fd.append('source', activeTab);
    fd.append('subject_name', form.querySelector('[name="subject_name"]').value);

    if (activeTab === 'upload') {
      const f = videoInput.files[0];
      if (!f) return;
      fd.append('video', f, f.name);
    } else {
      if (!recordedBlob) return;
      fd.append('video', recordedBlob, `record_${Date.now()}.webm`);
    }

    statusBox.classList.remove('hidden', 'status-error');
    statusBox.innerHTML = '<span class="spinner"></span> Mengekstrak wajah & menjalankan model... (bisa 5–15 detik)';
    resultBox.classList.add('hidden');
    btnPredict.disabled = true;

    try {
      const res  = await fetch('/predict', { method: 'POST', body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);

      statusBox.innerHTML = `✓ Selesai dalam ${data.processing_time}s`;
      renderResult(data);
    } catch (err) {
      statusBox.classList.add('status-error');
      statusBox.innerHTML = `❌ ${err.message}`;
    } finally {
      btnPredict.disabled = false;
    }
  });

function renderResult(data) {
    const meta = [
      ['Openness',          data.scores.openness,          '#7c3aed', 'Keterbukaan pengalaman baru'],
      ['Conscientiousness', data.scores.conscientiousness, '#0891b2', 'Ketelitian & disiplin'],
      ['Extraversion',      data.scores.extraversion,      '#ea580c', 'Tingkat sosialitas'],
      ['Agreeableness',     data.scores.agreeableness,     '#16a34a', 'Empati & kerjasama'],
      ['Neuroticism',       data.scores.neuroticism,       '#dc2626', 'Stabilitas emosi'],
    ];
    traitGrid.innerHTML = meta.map(([name, v, c, desc]) => `
      <div class="trait-card">
        <div class="trait-head">
          <span class="trait-name">${name}</span>
          <span class="trait-val">${v.toFixed(2)}%</span>  <!-- ✅ Ubah ke persen format -->
        </div>
        <div class="bar-wrap"><div class="bar-fill" style="width:${v.toFixed(1)}%; background:${c};"></div></div>
        <small class="muted">${desc}</small>
      </div>
    `).join('');
    detailLink.href = data.result_url;
    resultBox.classList.remove('hidden');
    resultBox.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
})();
