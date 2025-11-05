import type { Config } from "tailwindcss";

export default {
    content: [
        "./pages/**/*.{js,ts,jsx,tsx,mdx}",
        "./components/**/*.{js,ts,jsx,tsx,mdx}",
        "./app/**/*.{js,ts,jsx,tsx,mdx}",
    ],
    theme: {
        extend: {
            colors: {
                background: "var(--background)",
                foreground: "var(--foreground)",
                pastel: {
                    lavender: '#E6E6FA',
                    blue: '#B4D4FF',
                    mint: '#D4F1F4',
                    peach: '#FFE5D9',
                    coral: '#FFB5C5',
                    yellow: '#FFF4B7',
                }
            },
            fontFamily: {
                mono: ['Courier Prime', 'IBM Plex Mono', 'Courier New', 'monospace'],
            },
            backdropBlur: {
                glass: '16px',
            }
        },
    },
    plugins: [],
} satisfies Config;
