import { Compass } from "lucide-react";
import { useDocumentTitle } from "../components/layout/chrome.jsx";
import { ButtonLink, EmptyState } from "../components/ui/primitives.jsx";

export default function NotFoundPage() {
  useDocumentTitle("Page not found");
  return (
    <EmptyState
      icon={Compass}
      title="Page not found"
      description="The page you are looking for does not exist or has moved."
      action={<ButtonLink to="/dashboard">Go to the dashboard</ButtonLink>}
      className="mt-10"
    />
  );
}
