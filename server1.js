const deepspeech = require("deepspeech");
const fs = require("fs");
const path = require("path");
const ffmpeg = require("fluent-ffmpeg");
const ffmpegPath = require("@ffmpeg-installer/ffmpeg").path;

// Configure ffmpeg path
ffmpeg.setFfmpegPath(ffmpegPath);

// Load model and scorer paths
const modelPath = path.resolve("./deepspeech-0.9.3-models.pbmm");
const scorerPath = path.resolve("./deepspeech-0.9.3-models.scorer");

// Check model files
if (!fs.existsSync(modelPath) || !fs.existsSync(scorerPath)) {
  console.error("Model or scorer files not found");
  process.exit(1);
}

// Initialize DeepSpeech model
let model;
try {
  model = new deepspeech.Model(modelPath);
  model.enableExternalScorer(scorerPath);
} catch (error) {
  console.error("Error initializing DeepSpeech model:", error);
  process.exit(1);
}

// Function to calculate Word Error Rate (WER)
function calculateWER(reference, hypothesis) {
  const ref = reference.toLowerCase().split(" ");
  const hyp = hypothesis.toLowerCase().split(" ");

  // Create matrix
  const dp = Array(ref.length + 1)
    .fill(null)
    .map(() => Array(hyp.length + 1).fill(0));

  // Initialize first row and column
  for (let i = 0; i <= ref.length; i++) dp[i][0] = i;
  for (let j = 0; j <= hyp.length; j++) dp[0][j] = j;

  // Fill matrix
  for (let i = 1; i <= ref.length; i++) {
    for (let j = 1; j <= hyp.length; j++) {
      if (ref[i - 1] === hyp[j - 1]) {
        dp[i][j] = dp[i - 1][j - 1];
      } else {
        dp[i][j] = Math.min(
          dp[i - 1][j - 1] + 1, // substitution
          dp[i][j - 1] + 1, // insertion
          dp[i - 1][j] + 1 // deletion
        );
      }
    }
  }

  const wer = dp[ref.length][hyp.length] / ref.length;
  return wer;
}

// Function to calculate Word Accuracy
function calculateAccuracy(reference, hypothesis) {
  const wer = calculateWER(reference, hypothesis);
  return ((1 - wer) * 100).toFixed(2);
}

async function transcribeAudio(audioFilePath, referenceText = null) {
  const startTime = process.hrtime();
  let chunkCount = 0;
  let totalChunkProcessingTime = 0;

  // Create log files with timestamp
  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  const transcriptionFile = fs.createWriteStream(
    `transcription_${timestamp}.txt`
  );
  const performanceLog = fs.createWriteStream(`performance_${timestamp}.txt`);

  try {
    transcriptionFile.write("Starting transcription...\n");
    transcriptionFile.write(`Processing file: ${audioFilePath}\n\n`);

    const fileStats = fs.statSync(audioFilePath);
    transcriptionFile.write(
      `Input File Size: ${(fileStats.size / 1024 / 1024).toFixed(2)} MB\n\n`
    );

    const chunks = [];
    let currentBuffer = Buffer.alloc(0);
    const CHUNK_SIZE = 32000;

    await new Promise((resolve, reject) => {
      ffmpeg(audioFilePath)
        .toFormat("wav")
        .audioChannels(1)
        .audioFrequency(16000)
        .audioCodec("pcm_s16le")
        .on("error", (err) => {
          console.error("FFmpeg error:", err);
          transcriptionFile.write(`FFmpeg error: ${err}\n`);
          reject(err);
        })
        .on("end", () => {
          if (currentBuffer.length > 0) {
            const finalText = model.stt(currentBuffer);
            if (finalText.trim()) {
              transcriptionFile.write(`Final chunk: "${finalText}"\n`);
              chunkCount++;
            }
          }
          resolve();
        })
        .pipe()
        .on("data", (chunk) => {
          chunks.push(chunk);
          currentBuffer = Buffer.concat([currentBuffer, chunk]);

          while (currentBuffer.length >= CHUNK_SIZE) {
            const chunkToProcess = currentBuffer.slice(0, CHUNK_SIZE);
            currentBuffer = currentBuffer.slice(CHUNK_SIZE);

            try {
              const intermediateText = model.stt(chunkToProcess);
              if (intermediateText.trim()) {
                transcriptionFile.write(
                  `Chunk ${chunkCount + 1}: "${intermediateText}"\n`
                );
                chunkCount++;
              }
            } catch (err) {
              transcriptionFile.write(
                `Error processing chunk: ${err.message}\n`
              );
            }
          }
        });
    });

    // Process complete audio
    transcriptionFile.write("\n=== Complete Transcription ===\n");
    const completeBuffer = Buffer.concat(chunks);
    const finalResult = model.stt(completeBuffer);
    transcriptionFile.write(finalResult + "\n");
    transcriptionFile.write("===========================\n\n");

    // Calculate and log accuracy if reference text is provided
    if (referenceText) {
      const accuracy = calculateAccuracy(referenceText, finalResult);
      transcriptionFile.write("\nAccuracy Metrics:\n");
      transcriptionFile.write(`Reference text: "${referenceText}"\n`);
      transcriptionFile.write(`Transcribed text: "${finalResult}"\n`);
      transcriptionFile.write(`Word Accuracy: ${accuracy}%\n\n`);

      // Log performance metrics
      performanceLog.write("\nPerformance Metrics:\n");
      performanceLog.write(`Total Chunks: ${chunkCount}\n`);
      const endTime = process.hrtime(startTime);
      const totalTime = endTime[0] + endTime[1] / 1e9;
      performanceLog.write(
        `Total Processing Time: ${totalTime.toFixed(2)} seconds\n`
      );
      performanceLog.write(`Word Accuracy: ${accuracy}%\n`);
    }

    // Ensure files are properly written before closing
    await new Promise((resolve) => {
      transcriptionFile.write("Transcription completed\n", () => {
        transcriptionFile.end();
        performanceLog.end();
        resolve();
      });
    });

    return finalResult;
  } catch (error) {
    console.error("Error during transcription:", error);
    transcriptionFile.write(`Error: ${error.message}\n`);
    transcriptionFile.end();
    performanceLog.end();
    throw error;
  }
}

// Example usage with reference text
const audioFilePath = path.join(__dirname, "nicole.mp3");
const referenceText =
  "With a soft and whispery American accent I am the ideal choice for creating ASMR content, meditative guides, or adding an intimate feel to your narrative projects";

transcribeAudio(audioFilePath, referenceText)
  .then(() => {
    console.log("Transcription process completed");
  })
  .catch((error) => {
    console.error("Transcription failed:", error);
  });
