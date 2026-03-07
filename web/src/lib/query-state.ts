type QueryStateLike = {
  isLoading: boolean;
  error: unknown;
};

type QueryDataLike<T> = QueryStateLike & {
  data?: T;
};

type QueryEnvelopeLike<T> = QueryStateLike & {
  data?: {
    data: T;
  };
};

export interface ResolvedQueryData<T> extends QueryStateLike {
  data: T;
  usingMock: boolean;
}

export interface QueryNoticeState {
  loading: boolean;
  error: unknown;
  usingMock: boolean;
}

export function resolveQueryData<T>(query: QueryDataLike<T>, mockData: T): ResolvedQueryData<T> {
  const usingMock = !query.data;
  return {
    data: query.data ?? mockData,
    usingMock,
    isLoading: query.isLoading,
    error: query.error,
  };
}

export function resolveReportQueryData<T>(query: QueryEnvelopeLike<T>, mockData: T): ResolvedQueryData<T> {
  const usingMock = !query.data;
  return {
    data: query.data?.data ?? mockData,
    usingMock,
    isLoading: query.isLoading,
    error: query.error,
  };
}

export function toQueryNoticeState<T>(state: ResolvedQueryData<T>): QueryNoticeState {
  return {
    loading: state.isLoading,
    error: state.error,
    usingMock: state.usingMock,
  };
}

export function mergeQueryNoticeStates(...states: ReadonlyArray<QueryNoticeState>): QueryNoticeState {
  return {
    loading: states.some((state) => state.loading),
    error: states.find((state) => state.error)?.error ?? null,
    usingMock: states.some((state) => state.usingMock),
  };
}
