import candidates from "@/data/candidates.json";

export const runtime = "nodejs";

export function GET() {
  return Response.json({
    ok: true,
    as_of: candidates.as_of,
    strategy_version: candidates.strategy_version,
    source: candidates.source,
  });
}
