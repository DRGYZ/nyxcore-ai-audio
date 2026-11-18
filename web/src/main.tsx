import React from "react";
import ReactDOM from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter } from "react-router-dom";
import { App } from "./ui/App";
import { DesktopEntry } from "./ui/DesktopLibrary";
import "./index.css";

const queryClient = new QueryClient();
const desktop = document.querySelector('meta[name="nyxcore-desktop"][content="1"]') !== null;

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        {desktop ? <DesktopEntry renderApp={(session) => <App desktop desktopSession={session} />} /> : <App />}
      </BrowserRouter>
    </QueryClientProvider>
  </React.StrictMode>,
);
