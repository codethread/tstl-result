// export { Result, ok, Ok, err, Err, fromThrowable } from './result'
// export { ResultAsync, okAsync, errAsync, fromPromise, fromSafePromise } from './result-async'

interface ErrorConfig {
  withStackTrace: boolean;
}

const defaultErrorConfig: ErrorConfig = {
  withStackTrace: false,
};

interface NeverThrowError<T, E> {
  data:
    | {
        type: string;
        value: T;
      }
    | {
        type: string;
        value: E;
      };
  message: string;
  stack: string | undefined;
}

// Custom error object
// Context / discussion: https://github.com/supermacro/neverthrow/pull/215
const createNeverThrowError = <T, E>(
  message: string,
  result: Result<T, E>,
  config: ErrorConfig = defaultErrorConfig
): NeverThrowError<T, E> => {
  const data = result.isOk()
    ? { type: "Ok", value: result.value }
    : { type: "Err", value: result.error };

  const maybeStack = config.withStackTrace ? new Error().stack : undefined;

  return {
    data,
    message,
    stack: maybeStack,
  };
};

// Given a list of Results, this extracts all the different `T` types from that list
type ExtractOkTypes<T extends readonly Result<unknown, unknown>[]> = {
  [idx in keyof T]: T[idx] extends Result<infer U, unknown> ? U : never;
};

// Given a list of ResultAsyncs, this extracts all the different `T` types from that list
type ExtractOkAsyncTypes<T extends readonly ResultAsync<unknown, unknown>[]> = {
  [idx in keyof T]: T[idx] extends ResultAsync<infer U, unknown> ? U : never;
};

// Given a list of Results, this extracts all the different `E` types from that list
type ExtractErrTypes<T extends readonly Result<unknown, unknown>[]> = {
  [idx in keyof T]: T[idx] extends Result<unknown, infer E> ? E : never;
};

// Given a list of ResultAsyncs, this extracts all the different `E` types from that list
type ExtractErrAsyncTypes<T extends readonly ResultAsync<unknown, unknown>[]> =
  {
    [idx in keyof T]: T[idx] extends ResultAsync<unknown, infer E> ? E : never;
  };

type InferOkTypes<R> = R extends Result<infer T, unknown> ? T : never;
type InferErrTypes<R> = R extends Result<unknown, infer E> ? E : never;

type InferAsyncOkTypes<R> = R extends ResultAsync<infer T, unknown> ? T : never;
type InferAsyncErrTypes<R> = R extends ResultAsync<unknown, infer E>
  ? E
  : never;

const appendValueToEndOfList =
  <T>(value: T) =>
  (list: T[]): T[] =>
    [...list, value];

/**
 * Short circuits on the FIRST Err value that we find
 */
const combineResultList = <T, E>(
  resultList: readonly Result<T, E>[]
): Result<readonly T[], E> =>
  resultList.reduce(
    (acc, result) =>
      acc.isOk()
        ? result.isErr()
          ? err(result.error)
          : acc.map(appendValueToEndOfList(result.value))
        : acc,
    ok([]) as Result<T[], E>
  );

/* This is the typesafe version of Promise.all
 *
 * Takes a list of ResultAsync<T, E> and success if all inner results are Ok values
 * or fails if one (or more) of the inner results are Err values
 */
const combineResultAsyncList = <T, E>(
  asyncResultList: readonly ResultAsync<T, E>[]
): ResultAsync<readonly T[], E> =>
  ResultAsync.fromSafePromise(Promise.all(asyncResultList)).andThen(
    combineResultList
  ) as ResultAsync<T[], E>;

/**
 * Give a list of all the errors we find
 */
const combineResultListWithAllErrors = <T, E>(
  resultList: readonly Result<T, E>[]
): Result<readonly T[], E[]> =>
  resultList.reduce(
    (acc, result) =>
      result.isErr()
        ? acc.isErr()
          ? err([...acc.error, result.error])
          : err([result.error])
        : acc.isErr()
        ? acc
        : ok([...acc.value, result.value]),
    ok([]) as Result<T[], E[]>
  );

const combineResultAsyncListWithAllErrors = <T, E>(
  asyncResultList: readonly ResultAsync<T, E>[]
): ResultAsync<readonly T[], E[]> =>
  ResultAsync.fromSafePromise(Promise.all(asyncResultList)).andThen(
    combineResultListWithAllErrors
  ) as ResultAsync<T[], E[]>;

/**
 * Wraps a function with a try catch, creating a new function with the same
 * arguments but returning `Ok` if successful, `Err` if the function throws
 *
 * @param fn function to wrap with ok on success or err on failure
 * @param errorFn when an error is thrown, this will wrap the error result if provided
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function fromThrowable<Fn extends (...args: readonly any[]) => any, E>(
  fn: Fn,
  errorFn?: (e: unknown) => E
): (...args: Parameters<Fn>) => Result<ReturnType<Fn>, E> {
  return (...args) => {
    try {
      const result = fn(...args);
      return ok(result);
      // HACK any to resolve ts issue
    } catch (e: any) {
      return err(errorFn ? errorFn(e) : e);
    }
  };
}

export function combine<
  T extends readonly [Result<unknown, unknown>, ...Result<unknown, unknown>[]]
>(resultList: T): CombineResults<T>;
export function combine<T extends readonly Result<unknown, unknown>[]>(
  resultList: T
): CombineResults<T>;
export function combine<
  T extends readonly [Result<unknown, unknown>, ...Result<unknown, unknown>[]]
>(resultList: T): CombineResults<T> {
  return combineResultList(resultList) as CombineResults<T>;
}

export function combineWithAllErrors<
  T extends readonly [Result<unknown, unknown>, ...Result<unknown, unknown>[]]
>(resultList: T): CombineResultsWithAllErrorsArray<T>;
export function combineWithAllErrors<
  T extends readonly Result<unknown, unknown>[]
>(resultList: T): CombineResultsWithAllErrorsArray<T>;
export function combineWithAllErrors<
  T extends readonly Result<unknown, unknown>[]
>(resultList: T): CombineResultsWithAllErrorsArray<T> {
  return combineResultListWithAllErrors(
    resultList
  ) as CombineResultsWithAllErrorsArray<T>;
}

export type Result<T, E> = Ok<T, E> | Err<T, E>;

export const ok = <T, E = never>(value: T): Ok<T, E> => new Ok(value);

export const err = <T = never, E = unknown>(err: E): Err<T, E> => new Err(err);

interface IResult<T, E> {
  /**
   * Used to check if a `Result` is an `OK`
   *
   * @returns `true` if the result is an `OK` variant of Result
   */
  isOk(): this is Ok<T, E>;

  /**
   * Used to check if a `Result` is an `Err`
   *
   * @returns `true` if the result is an `Err` variant of Result
   */
  isErr(): this is Err<T, E>;

  /**
   * Maps a `Result<T, E>` to `Result<U, E>`
   * by applying a function to a contained `Ok` value, leaving an `Err` value
   * untouched.
   *
   * @param f The function to apply an `OK` value
   * @returns the result of applying `f` or an `Err` untouched
   */
  map<A>(f: (t: T) => A): Result<A, E>;

  /**
   * Maps a `Result<T, E>` to `Result<T, F>` by applying a function to a
   * contained `Err` value, leaving an `Ok` value untouched.
   *
   * This function can be used to pass through a successful result while
   * handling an error.
   *
   * @param f a function to apply to the error `Err` value
   */
  mapErr<U>(f: (e: E) => U): Result<T, U>;

  /**
   * Similar to `map` Except you must return a new `Result`.
   *
   * This is useful for when you need to do a subsequent computation using the
   * inner `T` value, but that computation might fail.
   * Additionally, `andThen` is really useful as a tool to flatten a
   * `Result<Result<A, E2>, E1>` into a `Result<A, E2>` (see example below).
   *
   * @param f The function to apply to the current value
   */
  andThen<R extends Result<unknown, unknown>>(
    f: (t: T) => R
  ): Result<InferOkTypes<R>, InferErrTypes<R> | E>;
  andThen<U, F>(f: (t: T) => Result<U, F>): Result<U, E | F>;

  /**
   * Takes an `Err` value and maps it to a `Result<T, SomeNewType>`.
   *
   * This is useful for error recovery.
   *
   *
   * @param f  A function to apply to an `Err` value, leaving `Ok` values
   * untouched.
   */
  orElse<R extends Result<unknown, unknown>>(
    f: (e: E) => R
  ): Result<T, InferErrTypes<R>>;
  orElse<A>(f: (e: E) => Result<T, A>): Result<T, A>;

  /**
   * Similar to `map` Except you must return a new `Result`.
   *
   * This is useful for when you need to do a subsequent async computation using
   * the inner `T` value, but that computation might fail. Must return a ResultAsync
   *
   * @param f The function that returns a `ResultAsync` to apply to the current
   * value
   */
  asyncAndThen<U, F>(f: (t: T) => ResultAsync<U, F>): ResultAsync<U, E | F>;

  /**
   * Maps a `Result<T, E>` to `ResultAsync<U, E>`
   * by applying an async function to a contained `Ok` value, leaving an `Err`
   * value untouched.
   *
   * @param f An async function to apply an `OK` value
   */
  asyncMap<U>(f: (t: T) => Promise<U>): ResultAsync<U, E>;

  /**
   * Unwrap the `Ok` value, or return the default if there is an `Err`
   *
   * @param v the default value to return if there is an `Err`
   */
  unwrapOr<A>(v: A): T | A;

  /**
   *
   * Given 2 functions (one for the `Ok` variant and one for the `Err` variant)
   * execute the function that matches the `Result` variant.
   *
   * Match callbacks do not necessitate to return a `Result`, however you can
   * return a `Result` if you want to.
   *
   * `match` is like chaining `map` and `mapErr`, with the distinction that
   * with `match` both functions must have the same return type.
   *
   * @param ok
   * @param err
   */
  match<A>(ok: (t: T) => A, err: (e: E) => A): A;

  /**
   * **This method is unsafe, and should only be used in a test environments**
   *
   * Takes a `Result<T, E>` and returns a `T` when the result is an `Ok`, otherwise it throws a custom object.
   *
   * @param config
   */
  _unsafeUnwrap(config?: ErrorConfig): T;

  /**
   * **This method is unsafe, and should only be used in a test environments**
   *
   * takes a `Result<T, E>` and returns a `E` when the result is an `Err`,
   * otherwise it throws a custom object.
   *
   * @param config
   */
  _unsafeUnwrapErr(config?: ErrorConfig): E;
}

export class Ok<T, E> implements IResult<T, E> {
  constructor(readonly value: T) {}

  isOk(): this is Ok<T, E> {
    return true;
  }

  isErr(): this is Err<T, E> {
    return !this.isOk();
  }

  map<A>(f: (t: T) => A): Result<A, E> {
    return ok(f(this.value));
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  mapErr<U>(_f: (e: E) => U): Result<T, U> {
    return ok(this.value);
  }

  andThen<R extends Result<unknown, unknown>>(
    f: (t: T) => R
  ): Result<InferOkTypes<R>, InferErrTypes<R> | E>;
  andThen<U, F>(f: (t: T) => Result<U, F>): Result<U, E | F>;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/explicit-module-boundary-types
  andThen(f: any): any {
    return f(this.value);
  }

  orElse<R extends Result<unknown, unknown>>(
    _f: (e: E) => R
  ): Result<T, InferErrTypes<R>>;
  orElse<A>(_f: (e: E) => Result<T, A>): Result<T, A>;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/explicit-module-boundary-types
  orElse(_f: any): any {
    return ok(this.value);
  }

  asyncAndThen<U, F>(f: (t: T) => ResultAsync<U, F>): ResultAsync<U, E | F> {
    return f(this.value);
  }

  asyncMap<U>(f: (t: T) => Promise<U>): ResultAsync<U, E> {
    return ResultAsync.fromSafePromise(f(this.value));
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  unwrapOr<A>(_v: A): T | A {
    return this.value;
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  match<A>(ok: (t: T) => A, _err: (e: E) => A): A {
    return ok(this.value);
  }

  _unsafeUnwrap(_?: ErrorConfig): T {
    return this.value;
  }

  _unsafeUnwrapErr(config?: ErrorConfig): E {
    throw createNeverThrowError(
      "Called `_unsafeUnwrapErr` on an Ok",
      this,
      config
    );
  }
}

export class Err<T, E> implements IResult<T, E> {
  constructor(readonly error: E) {}

  isOk(): this is Ok<T, E> {
    return false;
  }

  isErr(): this is Err<T, E> {
    return !this.isOk();
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  map<A>(_f: (t: T) => A): Result<A, E> {
    return err(this.error);
  }

  mapErr<U>(f: (e: E) => U): Result<T, U> {
    return err(f(this.error));
  }

  andThen<R extends Result<unknown, unknown>>(
    _f: (t: T) => R
  ): Result<InferOkTypes<R>, InferErrTypes<R> | E>;
  andThen<U, F>(_f: (t: T) => Result<U, F>): Result<U, E | F>;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/explicit-module-boundary-types
  andThen(_f: any): any {
    return err(this.error);
  }

  orElse<R extends Result<unknown, unknown>>(
    f: (e: E) => R
  ): Result<T, InferErrTypes<R>>;
  orElse<A>(f: (e: E) => Result<T, A>): Result<T, A>;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/explicit-module-boundary-types
  orElse(f: any): any {
    return f(this.error);
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  asyncAndThen<U, F>(_f: (t: T) => ResultAsync<U, F>): ResultAsync<U, E | F> {
    return errAsync<U, E>(this.error);
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  asyncMap<U>(_f: (t: T) => Promise<U>): ResultAsync<U, E> {
    return errAsync<U, E>(this.error);
  }

  unwrapOr<A>(v: A): T | A {
    return v;
  }

  match<A>(_ok: (t: T) => A, err: (e: E) => A): A {
    return err(this.error);
  }

  _unsafeUnwrap(config?: ErrorConfig): T {
    throw createNeverThrowError(
      "Called `_unsafeUnwrap` on an Err",
      this,
      config
    );
  }

  _unsafeUnwrapErr(_?: ErrorConfig): E {
    return this.error;
  }
}

//#region Combine - Types

// This is a helper type to prevent infinite recursion in typing rules.
//
// Use this with your `depth` variable in your types.
type Prev = [
  never,
  0,
  1,
  2,
  3,
  4,
  5,
  6,
  7,
  8,
  9,
  10,
  11,
  12,
  13,
  14,
  15,
  16,
  17,
  18,
  19,
  20,
  21,
  22,
  23,
  24,
  25,
  26,
  27,
  28,
  29,
  30,
  31,
  32,
  33,
  34,
  35,
  36,
  37,
  38,
  39,
  40,
  41,
  42,
  43,
  44,
  45,
  46,
  47,
  48,
  49,
  ...0[]
];

// Collects the results array into separate tuple array.
//
// T         - The array of the results
// Collected - The collected tuples.
// Depth     - The maximum depth.
type CollectResults<
  T,
  Collected extends unknown[] = [],
  Depth extends number = 50
> = [Depth] extends [never]
  ? []
  : T extends [infer H, ...infer Rest]
  ? // And test whether the head of the list is a result
    H extends Result<infer L, infer R>
    ? // Continue collecting...
      CollectResults<
        // the rest of the elements
        Rest,
        // The collected
        [...Collected, [L, R]],
        // and one less of the current depth
        Prev[Depth]
      >
    : never // Impossible
  : Collected;

// Transposes an array
//
// A          - The array source
// Transposed - The collected transposed array
// Depth      - The maximum depth.
type Transpose<
  A,
  Transposed extends unknown[][] = [],
  Depth extends number = 10
> = A extends [infer T, ...infer Rest]
  ? T extends [infer L, infer R]
    ? Transposed extends [infer PL, infer PR]
      ? PL extends unknown[]
        ? PR extends unknown[]
          ? Transpose<Rest, [[...PL, L], [...PR, R]], Prev[Depth]>
          : never
        : never
      : Transpose<Rest, [[L], [R]], Prev[Depth]>
    : Transposed
  : Transposed;

// Combines the both sides of the array of the results into a tuple of the
// union of the ok types and the union of the err types.
//
// T     - The array of the results
// Depth - The maximum depth.
type Combine<T, Depth extends number = 5> = Transpose<
  CollectResults<T>,
  [],
  Depth
> extends [infer L, infer R]
  ? [UnknownMembersToNever<L>, UnknownMembersToNever<R>]
  : Transpose<CollectResults<T>, [], Depth> extends []
  ? [[], []]
  : never;

// Deduplicates the result, as the result type is a union of Err and Ok types.
type Dedup<T> = T extends Result<infer RL, infer RR>
  ? [unknown] extends [RL]
    ? Err<RL, RR>
    : Ok<RL, RR>
  : T;

// Given a union, this gives the array of the union members.
type MemberListOf<T> = (
  (T extends unknown ? (t: T) => T : never) extends infer U
    ? (U extends unknown ? (u: U) => unknown : never) extends (
        v: infer V
      ) => unknown
      ? V
      : never
    : never
) extends (_: unknown) => infer W
  ? [...MemberListOf<Exclude<T, W>>, W]
  : [];

// Converts an empty array to never.
//
// The second type parameter here will affect how to behave to `never[]`s.
// If a precise type is required, pass `1` here so that it will resolve
// a literal array such as `[ never, never ]`. Otherwise, set `0` or the default
// type value will cause this to resolve the arrays containing only `never`
// items as `never` only.
type EmptyArrayToNever<T, NeverArrayToNever extends number = 0> = T extends []
  ? never
  : NeverArrayToNever extends 1
  ? T extends [never, ...infer Rest]
    ? [EmptyArrayToNever<Rest>] extends [never]
      ? never
      : T
    : T
  : T;

// Converts the `unknown` items of an array to `never`s.
type UnknownMembersToNever<T> = T extends [infer H, ...infer R]
  ? [[unknown] extends [H] ? never : H, ...UnknownMembersToNever<R>]
  : T;

// Gets the member type of the array or never.
type MembersToUnion<T> = T extends unknown[] ? T[number] : never;

// Checks if the given type is a literal array.
type IsLiteralArray<T> = T extends { length: infer L }
  ? L extends number
    ? number extends L
      ? 0
      : 1
    : 0
  : 0;

// Traverses an array of results and returns a single result containing
// the oks and errs union-ed/combined.
type Traverse<T, Depth extends number = 5> = Combine<T, Depth> extends [
  infer Oks,
  infer Errs
]
  ? Result<EmptyArrayToNever<Oks, 1>, MembersToUnion<Errs>>
  : never;

// Traverses an array of results and returns a single result containing
// the oks combined and the array of errors combined.
type TraverseWithAllErrors<T, Depth extends number = 5> = Combine<
  T,
  Depth
> extends [infer Oks, infer Errs]
  ? Result<EmptyArrayToNever<Oks>, EmptyArrayToNever<Errs>>
  : never;

// Combines the array of results into one result.
type CombineResults<T extends readonly Result<unknown, unknown>[]> =
  IsLiteralArray<T> extends 1
    ? Traverse<T>
    : Result<ExtractOkTypes<T>, ExtractErrTypes<T>[number]>;

// Combines the array of results into one result with all errors.
type CombineResultsWithAllErrorsArray<
  T extends readonly Result<unknown, unknown>[]
> = IsLiteralArray<T> extends 1
  ? TraverseWithAllErrors<T>
  : Result<ExtractOkTypes<T>, ExtractErrTypes<T>[number][]>;

//#endregion
export class ResultAsync<T, E> implements PromiseLike<Result<T, E>> {
  private _promise: Promise<Result<T, E>>;

  constructor(res: Promise<Result<T, E>>) {
    this._promise = res;
  }

  static fromSafePromise<T, E = never>(
    promise: PromiseLike<T>
  ): ResultAsync<T, E>;
  static fromSafePromise<T, E = never>(promise: Promise<T>): ResultAsync<T, E> {
    const newPromise = promise.then((value: T) => new Ok<T, E>(value));

    return new ResultAsync(newPromise);
  }

  static fromPromise<T, E>(
    promise: PromiseLike<T>,
    errorFn: (e: unknown) => E
  ): ResultAsync<T, E>;
  static fromPromise<T, E>(
    promise: Promise<T>,
    errorFn: (e: unknown) => E
  ): ResultAsync<T, E> {
    const newPromise = promise
      .then((value: T) => new Ok<T, E>(value))
      .catch((e) => new Err<T, E>(errorFn(e)));

    return new ResultAsync(newPromise);
  }

  static combine<
    T extends readonly [
      ResultAsync<unknown, unknown>,
      ...ResultAsync<unknown, unknown>[]
    ]
  >(asyncResultList: T): CombineResultAsyncs<T>;
  static combine<T extends readonly ResultAsync<unknown, unknown>[]>(
    asyncResultList: T
  ): CombineResultAsyncs<T>;
  static combine<T extends readonly ResultAsync<unknown, unknown>[]>(
    asyncResultList: T
  ): CombineResultAsyncs<T> {
    return combineResultAsyncList(
      asyncResultList
    ) as unknown as CombineResultAsyncs<T>;
  }

  static combineWithAllErrors<
    T extends readonly [
      ResultAsync<unknown, unknown>,
      ...ResultAsync<unknown, unknown>[]
    ]
  >(asyncResultList: T): CombineResultsWithAllErrorsArrayAsync<T>;
  static combineWithAllErrors<
    T extends readonly ResultAsync<unknown, unknown>[]
  >(asyncResultList: T): CombineResultsWithAllErrorsArrayAsync<T>;
  static combineWithAllErrors<
    T extends readonly ResultAsync<unknown, unknown>[]
  >(asyncResultList: T): CombineResultsWithAllErrorsArrayAsync<T> {
    return combineResultAsyncListWithAllErrors(
      asyncResultList
    ) as CombineResultsWithAllErrorsArrayAsync<T>;
  }

  map<A>(f: (t: T) => A | Promise<A>): ResultAsync<A, E> {
    return new ResultAsync(
      this._promise.then(async (res: Result<T, E>) => {
        if (res.isErr()) {
          return new Err<A, E>(res.error);
        }

        return new Ok<A, E>(await f(res.value));
      })
    );
  }

  mapErr<U>(f: (e: E) => U | Promise<U>): ResultAsync<T, U> {
    return new ResultAsync(
      this._promise.then(async (res: Result<T, E>) => {
        if (res.isOk()) {
          return new Ok<T, U>(res.value);
        }

        return new Err<T, U>(await f(res.error));
      })
    );
  }

  andThen<R extends Result<unknown, unknown>>(
    f: (t: T) => R
  ): ResultAsync<InferOkTypes<R>, InferErrTypes<R> | E>;
  andThen<R extends ResultAsync<unknown, unknown>>(
    f: (t: T) => R
  ): ResultAsync<InferAsyncOkTypes<R>, InferAsyncErrTypes<R> | E>;
  andThen<U, F>(
    f: (t: T) => Result<U, F> | ResultAsync<U, F>
  ): ResultAsync<U, E | F>;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/explicit-module-boundary-types
  andThen(f: any): any {
    return new ResultAsync(
      this._promise.then((res) => {
        if (res.isErr()) {
          return new Err<never, E>(res.error);
        }

        const newValue = f(res.value);
        return newValue instanceof ResultAsync ? newValue._promise : newValue;
      })
    );
  }

  orElse<R extends Result<T, unknown>>(
    f: (e: E) => R
  ): ResultAsync<T, InferErrTypes<R>>;
  orElse<R extends ResultAsync<T, unknown>>(
    f: (e: E) => R
  ): ResultAsync<T, InferAsyncErrTypes<R>>;
  orElse<A>(f: (e: E) => Result<T, A> | ResultAsync<T, A>): ResultAsync<T, A>;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/explicit-module-boundary-types
  orElse(f: any): any {
    return new ResultAsync(
      this._promise.then(async (res: Result<T, E>) => {
        if (res.isErr()) {
          return f(res.error);
        }

        return new Ok<T, unknown>(res.value);
      })
    );
  }

  match<A>(ok: (t: T) => A, _err: (e: E) => A): Promise<A> {
    return this._promise.then((res) => res.match(ok, _err));
  }

  unwrapOr<A>(t: A): Promise<T | A> {
    return this._promise.then((res) => res.unwrapOr(t));
  }

  // Makes ResultAsync implement PromiseLike<Result>
  then<A, B>(
    successCallback?: (res: Result<T, E>) => A | PromiseLike<A>,
    failureCallback?: (reason: unknown) => B | PromiseLike<B>
  ): PromiseLike<A | B> {
    return this._promise.then(successCallback, failureCallback);
  }
}

export const okAsync = <T, E = never>(value: T): ResultAsync<T, E> =>
  new ResultAsync(Promise.resolve(new Ok<T, E>(value)));

export const errAsync = <T = never, E = unknown>(err: E): ResultAsync<T, E> =>
  new ResultAsync(Promise.resolve(new Err<T, E>(err)));

export const fromPromise = ResultAsync.fromPromise;
export const fromSafePromise = ResultAsync.fromSafePromise;

// Combines the array of async results into one result.
type CombineResultAsyncs<T extends readonly ResultAsync<unknown, unknown>[]> =
  IsLiteralArray<T> extends 1
    ? TraverseAsync<UnwrapAsync<T>>
    : ResultAsync<ExtractOkAsyncTypes<T>, ExtractErrAsyncTypes<T>[number]>;

// Combines the array of async results into one result with all errors.
type CombineResultsWithAllErrorsArrayAsync<
  T extends readonly ResultAsync<unknown, unknown>[]
> = IsLiteralArray<T> extends 1
  ? TraverseWithAllErrorsAsync<UnwrapAsync<T>>
  : ResultAsync<ExtractOkAsyncTypes<T>, ExtractErrAsyncTypes<T>[number][]>;

// Unwraps the inner `Result` from a `ResultAsync` for all elements.
type UnwrapAsync<T> = IsLiteralArray<T> extends 1
  ? Writable<T> extends [infer H, ...infer Rest]
    ? H extends PromiseLike<infer HI>
      ? HI extends Result<unknown, unknown>
        ? [Dedup<HI>, ...UnwrapAsync<Rest>]
        : never
      : never
    : []
  : // If we got something too general such as ResultAsync<X, Y>[] then we
  // simply need to map it to ResultAsync<X[], Y[]>. Yet `ResultAsync`
  // itself is a union therefore it would be enough to cast it to Ok.
  T extends Array<infer A>
  ? A extends PromiseLike<infer HI>
    ? HI extends Result<infer L, infer R>
      ? Ok<L, R>[]
      : never
    : never
  : never;

// Traverse through the tuples of the async results and create one
// `ResultAsync` where the collected tuples are merged.
type TraverseAsync<T, Depth extends number = 5> = IsLiteralArray<T> extends 1
  ? Combine<T, Depth> extends [infer Oks, infer Errs]
    ? ResultAsync<EmptyArrayToNever<Oks>, MembersToUnion<Errs>>
    : never
  : // The following check is important if we somehow reach to the point of
  // checking something similar to ResultAsync<X, Y>[]. In this case we don't
  // know the length of the elements, therefore we need to traverse the X and Y
  // in a way that the result should contain X[] and Y[].
  T extends Array<infer I>
  ? // The MemberListOf<I> here is to include all possible types. Therefore
    // if we face (ResultAsync<X, Y> | ResultAsync<A, B>)[] this type should
    // handle the case.
    Combine<MemberListOf<I>, Depth> extends [infer Oks, infer Errs]
    ? // The following `extends unknown[]` checks are just to satisfy the TS.
      // we already expect them to be an array.
      Oks extends unknown[]
      ? Errs extends unknown[]
        ? ResultAsync<
            EmptyArrayToNever<Oks[number][]>,
            MembersToUnion<Errs[number][]>
          >
        : ResultAsync<EmptyArrayToNever<Oks[number][]>, Errs>
      : // The rest of the conditions are to satisfy the TS and support
      // the edge cases which are not really expected to happen.
      Errs extends unknown[]
      ? ResultAsync<Oks, MembersToUnion<Errs[number][]>>
      : ResultAsync<Oks, Errs>
    : never
  : never;

// This type is similar to the `TraverseAsync` while the errors are also
// collected in order. For the checks/conditions made here, see that type
// for the documentation.
type TraverseWithAllErrorsAsync<
  T,
  Depth extends number = 5
> = IsLiteralArray<T> extends 1
  ? Combine<T, Depth> extends [infer Oks, infer Errs]
    ? ResultAsync<EmptyArrayToNever<Oks>, EmptyArrayToNever<Errs>>
    : never
  : Writable<T> extends Array<infer I>
  ? Combine<MemberListOf<I>, Depth> extends [infer Oks, infer Errs]
    ? Oks extends unknown[]
      ? Errs extends unknown[]
        ? ResultAsync<
            EmptyArrayToNever<Oks[number][]>,
            EmptyArrayToNever<Errs[number][]>
          >
        : ResultAsync<EmptyArrayToNever<Oks[number][]>, Errs>
      : Errs extends unknown[]
      ? ResultAsync<Oks, EmptyArrayToNever<Errs[number][]>>
      : ResultAsync<Oks, Errs>
    : never
  : never;

// Converts a reaodnly array into a writable array
type Writable<T> = T extends ReadonlyArray<unknown> ? [...T] : T;
