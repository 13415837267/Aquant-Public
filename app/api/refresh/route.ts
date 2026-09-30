export const runtime = "nodejs";

export async function GET(request: Request) {
  const cronSecret = process.env.CRON_SECRET;
  const supplied = request.headers.get("authorization")?.replace(/^Bearer\s+/i, "");
  const isCron = request.headers.get("user-agent")?.includes("vercel-cron/1.0");
  if (cronSecret && supplied !== cronSecret && !isCron) {
    return Response.json({ ok: false, error: "unauthorized" }, { status: 401 });
  }
  return Response.json({
    ok: true,
    message: "数据更新由 GitHub Actions Python pipeline 执行。",
  });
}
