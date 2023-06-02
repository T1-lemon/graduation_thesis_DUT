import { Box, LinearProgress } from '@mui/material'
import React, { useEffect, useState } from 'react'

const styles = {
  container: {
    width: '100%',
    position: 'fixed',
    top: 0,
    right: 0,
    left: 0,
    zIndex: 20
  },
  progress: {
    height: '6px',
    '& .MuiLinearProgress-bar': {
      backgroundColor: '#178f73'
    }
  }
}

type EventListener = {
  [key: string]: any
}

const eventlisteners: EventListener = {}
export const ProgressListener = {
  on: (eventName: string, funcCall: () => void) => {
    if (!eventName || !funcCall) {
      return
    }
    eventlisteners[eventName] = funcCall
  },
  off: (eventName: string) => {
    if (!eventName || !eventlisteners[eventName]) {
      return
    }
    delete eventlisteners[eventName]
  },
  emit: (eventName: string) => {
    if (!eventName || !eventlisteners[eventName]) {
      return
    }
    eventlisteners[eventName]()
  }
}

export default function Progress() {
  const [state, setState] = useState(false)

  useEffect(() => {
    ProgressListener.on('start', () => setState(true))
    ProgressListener.on('stop', () => setState(false))

    return () => {
      ProgressListener.off('start')
      ProgressListener.off('stop')
    }
  }, [])

  return (
    <Box sx={styles.container}>
      {state && <LinearProgress color='success' sx={styles.progress} />}
    </Box>
  )
}
