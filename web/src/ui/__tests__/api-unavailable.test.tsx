import { describe, it, expect, vi } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ApiUnavailableState } from "../feedback";
import { renderWithProviders } from "../../test/test-utils";
import * as hooks from "../../lib/hooks";

describe("ApiUnavailableState", () => {
  it("renders truthful offline setup instructions and commands without fake library content", () => {
    renderWithProviders(<ApiUnavailableState contextLabel="Overview" />);

    // Verify header and context
    expect(screen.getByRole("heading", { name: /local api is not connected • overview/i })).toBeInTheDocument();

    // Verify truthful uvicorn startup command is present
    expect(
      screen.getAllByText((content) =>
        content.includes("uvicorn nyxcore.webapi.app:app --reload --host 127.0.0.1 --port 8000")
      ).length
    ).toBeGreaterThanOrEqual(1);

    // Verify demo library setup command is present
    expect(
      screen.getByText((content) =>
        content.includes("demo/create_demo_library.py")
      )
    ).toBeInTheDocument();

    // Verify truthful local message
    expect(
      screen.getByText(/nyxcore operates locally\. no cloud services or external network requests are made\./i)
    ).toBeInTheDocument();

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
