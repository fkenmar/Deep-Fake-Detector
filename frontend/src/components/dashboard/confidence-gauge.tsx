import { cn } from "@/lib/utils";

interface ConfidenceGaugeProps {
  confidence: number;
  label: "Realism" | "Deepfake";
}

export function ConfidenceGauge({ confidence, label }: ConfidenceGaugeProps) {
  const isDeepfake = label === "Deepfake";

  return (
    <div className="bg-surface-elevated border border-border rounded-xl p-5">
      <div className="flex items-baseline justify-between mb-3">
        <div>
          <p className="text-xs font-medium text-on-surface-muted uppercase tracking-wider mb-1">
            Classification
          </p>
          <p
            className={cn(
              "text-xl font-bold",
              isDeepfake ? "text-danger" : "text-success"
            )}
          >
            {isDeepfake ? "DEEPFAKE" : "AUTHENTIC"}
          </p>
        </div>
        <div className="text-right">
          <p className="text-2xl font-bold text-on-surface font-mono">
            {confidence}%
          </p>
          <p className="text-xs text-on-surface-faint font-mono">
            softmax
          </p>
        </div>
      </div>

      <div className="relative h-3 bg-surface-sunken rounded-full overflow-hidden">
        <div
          className={cn(
            "absolute h-full rounded-full transition-all duration-700",
            isDeepfake ? "bg-danger" : "bg-success"
          )}
          style={{ width: `${confidence}%` }}
        />
      </div>

      <p className="text-xs text-on-surface-faint mt-2">
        Probability assigned by the classifier to its selected label.
      </p>
    </div>
  );
}
