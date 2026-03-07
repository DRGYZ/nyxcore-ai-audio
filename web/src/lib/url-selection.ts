import { useCallback, useEffect, useMemo } from "react";
import { useSearchParams } from "react-router-dom";

export function useUrlBackedSelection<TKey extends string, T extends Record<TKey, string>>({
  items,
  fallbackItems,
  param,
  idKey,
}: {
  items: readonly T[];
  fallbackItems?: readonly T[];
  param: string;
  idKey: TKey;
}) {
  const [searchParams, setSearchParams] = useSearchParams();
  const selectedId = searchParams.get(param);
  const preferredItems = fallbackItems ?? items;

  const selected = useMemo(() => {
    if (selectedId) {
      const exact = items.find((item) => item[idKey] === selectedId);
      if (exact) {
        return exact;
      }
    }
    return preferredItems[0] ?? items[0];
  }, [idKey, items, preferredItems, selectedId]);

  useEffect(() => {
    if (!items.length) {
      if (!selectedId) {
        return;
      }
      const next = new URLSearchParams(searchParams);
      next.delete(param);
      setSearchParams(next, { replace: true });
      return;
    }

    const nextId = selected?.[idKey];
    if (!nextId || nextId === selectedId) {
      return;
    }

    const next = new URLSearchParams(searchParams);
    next.set(param, nextId);
    setSearchParams(next, { replace: true });
  }, [idKey, items.length, param, searchParams, selected, selectedId, setSearchParams]);

  const selectById = useCallback(
    (nextId: string) => {
      const next = new URLSearchParams(searchParams);
      next.set(param, nextId);
      setSearchParams(next);
    },
    [param, searchParams, setSearchParams],
  );

  return {
    selectedId,
    selected,
    selectById,
  };
}
