using System;
using System.Runtime.InteropServices;
public static class WorldMemory {
    [StructLayout(LayoutKind.Sequential)]
    public struct Info {
        public uint cb;
        public UIntPtr CommitTotal, CommitLimit, CommitPeak, PhysicalTotal, PhysicalAvailable;
        public UIntPtr SystemCache, KernelTotal, KernelPaged, KernelNonpaged, PageSize;
        public uint HandleCount, ProcessCount, ThreadCount;
    }
    [DllImport("psapi.dll", SetLastError=true)]
    private static extern bool GetPerformanceInfo(out Info i, uint size);
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern bool GetExitCodeProcess(IntPtr process, out uint code);
    [DllImport("kernel32.dll", SetLastError=true)]
    private static extern bool SetProcessWorkingSetSizeEx(IntPtr process, UIntPtr min, UIntPtr max, uint flags);
    [DllImport("kernel32.dll", SetLastError=true)]
    private static extern bool GetProcessWorkingSetSizeEx(IntPtr process, out UIntPtr min, out UIntPtr max, out uint flags);
    public static void LimitResidentMemory(IntPtr process, int gib) {
        ulong requested=(ulong)gib << 30;
        if (!SetProcessWorkingSetSizeEx(process, new UIntPtr(16u << 20), new UIntPtr(requested), 4))
            throw new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error());
        UIntPtr min, max;
        uint flags;
        if (!GetProcessWorkingSetSizeEx(process, out min, out max, out flags))
            throw new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error());
        if (max.ToUInt64()!=requested || (flags & 4)==0)
            throw new InvalidOperationException("Windows did not apply the requested resident-memory ceiling.");
    }
    public static ulong[] Sample() {
        Info i;
        if (!GetPerformanceInfo(out i, (uint)Marshal.SizeOf(typeof(Info))))
            throw new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error());
        ulong page = i.PageSize.ToUInt64();
        return new ulong[] {page*i.CommitTotal.ToUInt64(), page*i.CommitLimit.ToUInt64(), page*i.PhysicalAvailable.ToUInt64()};
    }
}