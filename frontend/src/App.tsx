import { AppShell } from "@/components/layout/app-shell";
import { TaskQueueSidebar } from "@/components/sidebar/task-queue-sidebar";
import { ImageUpload } from "@/components/upload/image-upload";
import { ModelDashboard } from "@/components/dashboard/model-dashboard";
import { useQueueStore } from "@/stores/queue-store";

function App() {
  const { jobs, activeJobId } = useQueueStore();
  const activeJob = jobs.find((j) => j.id === activeJobId);
  const showDashboard =
    activeJob &&
    (activeJob.status === "complete" ||
      (activeJob.status === "error" && activeJob.error?.partialResult));

  return (
    <AppShell sidebar={<TaskQueueSidebar />}>
      {showDashboard ? (
        <ModelDashboard job={activeJob} />
      ) : (
        <div className="flex items-center justify-center min-h-[calc(100vh-200px)]">
          <ImageUpload />
        </div>
      )}
    </AppShell>
  );
}

export default App;
