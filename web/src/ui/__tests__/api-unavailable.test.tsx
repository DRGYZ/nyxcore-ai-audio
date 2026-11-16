import { describe, it, expect, vi } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ApiUnavailableState } from "../feedback";
import { renderWithProviders } from "../../test/test-utils";
import * as hooks from "../../lib/hooks";

describe("ApiUnavailableState", () => {
  it("guides users through a synthetic backend, connection check, and first finding", async () => {
    const user = userEvent.setup();
    renderWithProviders(<ApiUnavailableState contextLabel="Overview" />);

    // Verify header and context
    expect(screen.getByRole("heading", { name: /local api is not connected • overview/i })).toBeInTheDocument();

    expect(screen.getByText("Start the backend with the synthetic library.")).toBeInTheDocument();
    expect(screen.getByText("Confirm the connection.")).toBeInTheDocument();
    expect(screen.getByText("Open Overview and inspect a finding.")).toBeInTheDocument();
    await user.click(screen.getByText("WSL / Bash commands"));
    const bash = screen.getByText(/export NYXCORE_WEB_MUSIC_DIR/, { selector: "code" });
    expect(bash).toBeVisible();
    expect(bash).toHaveTextContent('export NYXCORE_WEB_MUSIC_DIR="$PWD/demo/generated/sample-library"');
    await user.click(screen.getByText("Native PowerShell commands"));
    const powershell = screen.getByText(/\$env:NYXCORE_WEB_MUSIC_DIR/, { selector: "code" });
    expect(powershell).toBeVisible();
    expect(powershell).toHaveTextContent("python demo/create_demo_library.py");
    expect(powershell).toHaveTextContent("python -m uvicorn nyxcore.webapi.app:app --host 127.0.0.1 --port 8000");
    expect(powershell).not.toHaveTextContent("export");

    // Verify fake library content is NOT displayed
    expect(screen.queryByText(/42,804 files scanned/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/blue train\.flac/i)).not.toBeInTheDocument();
  });

  it("triggers connection check when 'Check Connection' is clicked", async () => {
    const user = userEvent.setup();
    const mockCheck = vi.fn().mockResolvedValue(undefined);

    vi.spyOn(hooks, "useCheckConnection").mockReturnValue({
      checkConnection: mockCheck,
      checking: false,
    });

    renderWithProviders(<ApiUnavailableState onRetry={mockCheck} />);

    const checkButton = screen.getByRole("button", { name: /check connection/i });
    expect(checkButton).toBeEnabled();

    await user.click(checkButton);
    expect(mockCheck).toHaveBeenCalledTimes(1);
  });
});
