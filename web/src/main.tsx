import React from "react";
import ReactDOM from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter } from "react-router-dom";
import { App } from "./ui/App";
import "./index.css";

const queryClient = new QueryClient();
const desktop = new URLSearchParams(window.location.search).get("desktop") === "1";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <App desktop={desktop} />
      </BrowserRouter>
    </QueryClientProvider>
  </React.StrictMode>,
);
