import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Energy Management Puzzle",
  description: "Allocate limited power across systems",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">
        {children}
      </body>
    </html>
  );
}
