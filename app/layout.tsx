import type { Metadata } from "next";
import "./studio.css";

export const metadata: Metadata = {
  title: "ModelLab · Training studio",
  description: "A local studio for reproducible RunPod experiments.",
};
export default function Layout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
