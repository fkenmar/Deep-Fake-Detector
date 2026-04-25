export interface FaceResult {
  label: "Realism" | "Deepfake";
  confidence: number;
  bbox: [number, number, number, number] | null;
}

export interface PredictResponse {
  faces: FaceResult[];
  face_count: number;
  face_detected: boolean;
}

export type JobStatus =
  | "queued"
  | "uploading"
  | "detecting"
  | "analyzing"
  | "complete"
  | "error";

export interface AnalysisJob {
  id: string;
  fileName: string;
  thumbnail: string | null;
  status: JobStatus;
  currentStep: number;
  uploadProgress: number;
  queuePosition: number;
  eta: { min: number; max: number } | null;
  result: PredictResponse | null;
  error: {
    message: string;
    retryable: boolean;
    partialResult?: PredictResponse;
  } | null;
  createdAt: number;
}
