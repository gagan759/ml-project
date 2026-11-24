import { lazy, Suspense } from "react";
import { Route, Routes } from "react-router-dom";
import { MotionConfig } from "framer-motion";
import { DataProvider } from "./context/DataContext.jsx";
import { PredictionProvider } from "./context/PredictionContext.jsx";
import { ThemeProvider } from "./context/ThemeContext.jsx";
import AppLayout from "./components/layout/AppLayout.jsx";
import MarketingLayout from "./components/layout/MarketingLayout.jsx";
import PageLoader from "./components/ui/PageLoader.jsx";
import LandingPage from "./pages/LandingPage.jsx";

const DashboardPage = lazy(() => import("./pages/DashboardPage.jsx"));
const DiagnosePage = lazy(() => import("./pages/DiagnosePage.jsx"));
const LabPage = lazy(() => import("./pages/LabPage.jsx"));
const PerformancePage = lazy(() => import("./pages/PerformancePage.jsx"));
const ExplainabilityPage = lazy(() => import("./pages/ExplainabilityPage.jsx"));
const LibraryPage = lazy(() => import("./pages/LibraryPage.jsx"));
const DiseasePage = lazy(() => import("./pages/DiseasePage.jsx"));
const MethodologyPage = lazy(() => import("./pages/MethodologyPage.jsx"));
const DocsPage = lazy(() => import("./pages/DocsPage.jsx"));
const NotFoundPage = lazy(() => import("./pages/NotFoundPage.jsx"));

export default function App() {
  return (
    <MotionConfig reducedMotion="user">
      <ThemeProvider>
        <DataProvider>
          <PredictionProvider>
            <Suspense fallback={<PageLoader />}>
              <Routes>
                <Route element={<MarketingLayout />}>
                  <Route index element={<LandingPage />} />
                </Route>
                <Route element={<AppLayout />}>
                  <Route path="dashboard" element={<DashboardPage />} />
                  <Route path="diagnose" element={<DiagnosePage />} />
                  <Route path="lab" element={<LabPage />} />
                  <Route path="performance" element={<PerformancePage />} />
                  <Route path="explainability" element={<ExplainabilityPage />} />
                  <Route path="library" element={<LibraryPage />} />
                  <Route path="library/:key" element={<DiseasePage />} />
                  <Route path="methodology" element={<MethodologyPage />} />
                  <Route path="docs" element={<DocsPage />} />
                  <Route path="*" element={<NotFoundPage />} />
                </Route>
              </Routes>
            </Suspense>
          </PredictionProvider>
        </DataProvider>
      </ThemeProvider>
    </MotionConfig>
  );
}
