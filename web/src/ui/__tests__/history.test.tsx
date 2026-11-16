import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { HistoryPage } from "../pages/HistoryPage";
import { renderWithProviders } from "../../test/test-utils";
import { mockHistoryResponse } from "../../lib/mock-data";
import type { HistoryResponse } from "../../lib/types";

describe("History Page", () => {
  const [completed, reversed] = mockHistoryResponse.items;
  let history: HistoryResponse;
  let historyStatus: number;
  let connected: boolean;
  const fetchMock = vi.fn(async (input: string, options?: RequestInit) => {
    if (!connected) throw new TypeError("Failed to fetch");
    const path = new URL(input, "http://localhost").pathname;
    if (path === "/api/status") return Response.json({ status: "ok" });
    if (path === "/api/history") return Response.json(history, { status: historyStatus });
    if (path === "/api/history/" + completed.batch_id + "/restore" && options?.method === "POST") {
      const changedOperation = { ...completed.operations[0], undo_status: "ok" as const, undo_message: "undone" };
      history = {
        ...history,
        items: [
          { ...completed, lifecycle_state: "reversed", reversible: false, operations: [changedOperation] },
          reversed,
        ],
      };
      return Response.json({ batch_id: completed.batch_id, changed_operations: [changedOperation], reactivated_review_item_ids: [] });
    }
    throw new Error("Unexpected request to " + path);
  });

  beforeEach(() => {
    vi.clearAllMocks();
    history = mockHistoryResponse;
    historyStatus = 200;
    connected = true;
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => vi.unstubAllGlobals());

  it("exposes one canonical Reverse action and no separate Undo/Restore buttons", async () => {
    renderWithProviders(<HistoryPage />);

    expect(await screen.findByRole("button", { name: "Reverse Batch" })).toBeEnabled();

    expect(screen.queryByRole("button", { name: /^restore$/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /^undo$/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /^restore batch$/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /^undo batch$/i })).not.toBeInTheDocument();
  });

  it("opens reversal confirmation dialog, traps focus, and closes on cancel with focus restoration", async () => {
    const user = userEvent.setup();
    renderWithProviders(<HistoryPage />);

    const trigger = await screen.findByRole("button", { name: "Reverse Batch" });
    trigger.focus();
    expect(document.activeElement).toBe(trigger);

    await user.click(trigger);

    const modal = await screen.findByRole("dialog", { name: /confirm reversal/i });
    expect(modal).toBeInTheDocument();

    const cancelButton = screen.getByRole("button", { name: /cancel/i });
    await user.click(cancelButton);

    expect(screen.queryByRole("dialog", { name: /confirm reversal/i })).not.toBeInTheDocument();
  });

  it("shows completed and unresolved batches together and only enables settled reversal", async () => {
    const user = userEvent.setup();
    history = {
      ...history,
      meta: { ...history.meta, music_path: "C:\\Owner's Music", out_path: "C:\\Test Reports" },
      items: [
        {
          ...completed,
          batch_id: "batch-interrupted",
          lifecycle_state: "applying",
          recovery_required: true,
          operations: [{
            ...completed.operations[0],
            status: "applying",
            journal_state: "applying",
            original_path: "C:\\Owner's Music\\Interrupted.mp3",
            current_path: "C:\\Owner's Music\\Interrupted.mp3",
          }],
        },
        {
          ...completed,
          batch_id: "batch-pending-reversal",
          recovery_required: true,
          operations: [{ ...completed.operations[0], reversal_pending: true }],
        },
        ...history.items,
      ],
    };

    renderWithProviders(<HistoryPage />, { initialEntries: ["/history?batch=batch-interrupted"] });

    const interrupted = await screen.findByRole("row", { name: /batch-interrupted/ });
    expect(within(interrupted).getByText("Recovery Required")).toBeInTheDocument();
    expect(within(interrupted).getByRole("button", { name: "Reverse" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Reverse Batch" })).toBeDisabled();
    expect(screen.getByRole("heading", { name: "Recovery inspection required" })).toBeInTheDocument();
    expect(screen.getByText("Journal state: Applying")).toBeInTheDocument();
    expect(screen.getByText(/Interrupted\.mp3/)).toBeInTheDocument();
    expect(screen.getByText(
      "python -m nyxcore.cli recover-review-action 'batch-interrupted' --action inspect --music 'C:\\Owner''s Music' --out 'C:\\Test Reports'",
    )).toBeInTheDocument();
    expect(screen.getByText(/some states require manual intervention/i)).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /inspect batch batch-pending-reversal/i }));
    expect(screen.getByText("Reversal pending inspection")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Reverse Batch" })).toBeDisabled();

    expect(screen.getByText("Completed")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: new RegExp("Inspect batch " + completed.batch_id) }));
    expect(screen.getByRole("button", { name: "Reverse Batch" })).toBeEnabled();
    expect(screen.queryByRole("heading", { name: "Recovery inspection required" })).not.toBeInTheDocument();
  });

  it("runs a normal reversal and refreshes the batch outcome", async () => {
    const user = userEvent.setup();
    renderWithProviders(<HistoryPage />);
    await user.click(await screen.findByRole("button", { name: "Reverse Batch" }));
    const modal = screen.getByRole("dialog", { name: "Confirm Reversal" });
    await user.click(within(modal).getByRole("button", { name: "Reverse Batch" }));

    expect(await screen.findByText(/reversal completed for .*1 successful/i)).toBeInTheDocument();
    expect(screen.getByText("Latest Reversal Outcome")).toBeInTheDocument();
    const row = screen.getByRole("row", { name: new RegExp(completed.batch_id) });
    expect(await within(row).findByText("Reversed")).toBeInTheDocument();
    expect(within(row).getByRole("button", { name: "Reverse" })).toBeDisabled();
  });

  it("shows a History-specific error for HTTP 500 and retries without setup instructions", async () => {
    historyStatus = 500;
    const user = userEvent.setup();
    renderWithProviders(<HistoryPage />);

    expect(await screen.findByText(/history could not be loaded\. the local api is connected/i)).toBeInTheDocument();
    expect(screen.queryByText("Setup Instructions")).not.toBeInTheDocument();
    expect(screen.queryByText("No history batches recorded yet")).not.toBeInTheDocument();
    historyStatus = 200;
    await user.click(screen.getByRole("button", { name: "Retry History" }));
    expect(await screen.findByRole("button", { name: new RegExp("Inspect batch " + completed.batch_id) })).toBeInTheDocument();
  });

  it("shows connection guidance when the local API is unreachable", async () => {
    connected = false;
    renderWithProviders(<HistoryPage />);

    expect(await screen.findByRole("heading", { name: /local api is not connected • history/i }, { timeout: 2500 })).toBeInTheDocument();
    expect(screen.getByText("Setup Instructions")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Retry History" })).not.toBeInTheDocument();
  });
});
